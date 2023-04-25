import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

MAX_SPAN = 255

def _compute_relative_distance_matrix(query_length, key_length):
    """Computes a relative distance matrix between queries and keys.
    We assume that the queries and the keys are centered, i.e.,
    key_length = memory_flange + query_length + memory_flange.
    The function is based on the _generate_relative_positions_matrix function in
    common_attention.py of tensor2tensor codebase:
    https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1670
    Args:
      query_length: An integer, the length of queries.
      key_length: An integer, the length of keys.
    Returns:
      distance_matrix: A [query_length, key_length] tensor.
    Raises:
      ValueError: If (key_length - query_length) is odd, i.e., the assumption does
        not hold.
    """
    if (key_length - query_length) % 2:
        raise ValueError('Key_length should be query_length + 2 * memory_flange.')
    
    key_index = torch.arange(key_length)
    query_index = torch.arange(query_length) + (key_length - query_length) // 2
    distance_matrix = key_index[None, :] - query_index[:, None]
    
    # Shift the distance_matrix so that it is >= 0. Each entry of the
    # distance_matrix distance will index a relative positional embedding.
    distance_matrix = distance_matrix + MAX_SPAN - 1
    
    if query_length + (key_length - query_length) // 2 > MAX_SPAN:
        print(
            'Axial attention span is larger than MAX_SPAN. In this case, we use a '
            'single shared embedding for all positions beyond this relative '
            'distance. Please make sure, this behavior is intended.')
        distance_matrix = torch.clamp(distance_matrix, 0, MAX_SPAN * 2 - 2)
    
    return distance_matrix


class RelativePositionalEncoding(nn.Module):
    """Generates relative positional encoding.
    The function is based on the _generate_relative_positions_embeddings function
    in common_attention.py of tensor2tensor codebase:
    https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1691
    """

    def __init__(self, query_length, key_length, depth, name,
                 initialization_std=1.0, conv_kernel_weight_decay=0.0):
        """Initializes a relative position encoding layer.
        Args:
          query_length: An integer, the length of queries.
          key_length: An integer, the length of keys.
          depth: An integer, the number of embedding channels per head.
          name: A string, the name of the embedding.
          initialization_std: A float, the initialization std for the embedding.
          conv_kernel_weight_decay: A float, the weight decay for convolution
            kernels.
        Returns:
          output: A [query, key, depth] tensor, the relative positional
            encodings for each head and each query-key-pair.
        """
        super(RelativePositionalEncoding, self).__init__()
        self.name = name
        self._initializer = torch.nn.init.trunc_normal_(std=initialization_std)
        self._regularizer = nn.parameter.Parameter(torch.tensor(conv_kernel_weight_decay))

        self._relative_distance_matrix = _compute_relative_distance_matrix(query_length, key_length)
        self._embedding_shape = (MAX_SPAN * 2 - 1, depth)

        self._embeddings = nn.Parameter(torch.Tensor(*self._embedding_shape))
        self.reset_parameters()

    def reset_parameters(self):
        self._initializer(self._embeddings)

    def forward(self, inputs):
        """A forward pass that gathers the relative positional encoding."""
        del inputs
        # Gather the embeddings according to the relative distances.
        return torch.index_select(self._embeddings, 0, self._relative_distance_matrix.view(-1)).view(self._relative_distance_matrix.shape[0], self._relative_distance_matrix.shape[1], self._embeddings.shape[1])


class AxialAttention(nn.Module):
    """An axial-attention layer."""

    def __init__(self,
                 query_shape=129,
                 memory_flange=32,
                 total_key_depth=512,
                 total_value_depth=1024,
                 num_heads=8,
                 name='axial_attention',
                 use_query_rpe_similarity=True,
                 use_key_rpe_similarity=True,
                 use_content_similarity=True,
                 retrieve_value_rpe=True,
                 retrieve_value_content=True,
                 initialization_std_for_query_key_rpe=1.0,
                 initialization_std_for_value_rpe=1.0,
                 self_attention_activation='softmax',
                 bn_layer=nn.BatchNorm1d,
                 conv_kernel_weight_decay=0.0,
                 input_shape=(1, 1, 1)):

        super(AxialAttention, self).__init__()

        # Validate the attention similarity choices.
        if not any([
            use_content_similarity, use_key_rpe_similarity, use_query_rpe_similarity
        ]):
            raise ValueError(
                'Should use at least one similarity to compute attention.')

        # Validate the retrieve value choices.
        if not retrieve_value_content and not retrieve_value_rpe:
            raise ValueError('Should retrieve at least one of content or rpe.')

        if total_key_depth % num_heads:
            raise ValueError('Total_key_depth should be divisible by num_heads.')

        if total_value_depth % num_heads:
            raise ValueError('Total_value_depth should be divisible by num_heads.')

        self.query_shape = query_shape
        self.memory_flange = memory_flange
        self.total_key_depth = total_key_depth
        self.total_value_depth = total_value_depth
        self.num_heads = num_heads
        self.use_query_rpe_similarity = use_query_rpe_similarity
        self.use_key_rpe_similarity = use_key_rpe_similarity
        self.use_content_similarity = use_content_similarity
        self.retrieve_value_rpe = retrieve_value_rpe
        self.retrieve_value_content = retrieve_value_content
        self.initialization_std_for_query_key_rpe = (
            initialization_std_for_query_key_rpe)
        self.initialization_std_for_value_rpe = initialization_std_for_value_rpe
        self.self_attention_activation = self_attention_activation
        self.conv_kernel_weight_decay = conv_kernel_weight_decay

        self.batch_norm_qkv = bn_layer(total_key_depth * 2 + total_value_depth)
        self.batch_norm_similarity = bn_layer(total_key_depth)
        self.batch_norm_retrieved_output = bn_layer(total_value_depth)\

        self.key_depth_per_head = total_key_depth // num_heads
        self.attention_activate_fn = getattr(F, self_attention_activation)

        self.qkv_kernel = nn.Parameter(torch.Tensor(total_key_depth * 2 + total_value_depth))

        nn.init.trunc_normal_(self.qkv_kernel, std=total_key_depth**-0.5)

        if self.query_shape >= input_shape[1]:
          self.query_shape = input_shape[1]
          self.memory_flange = 0
        else:
          raise NotImplementedError('Local axial attention has not been '
                                    'implemented yet.')
        self.memory_shape = self.query_shape + 2 * self.memory_flange

        if self.use_query_rpe_similarity:
            self.query_rpe = RelativePositionalEncoding(
                self.query_shape,
                self.memory_shape,
                self.key_depth_per_head,
                initialization_std=self.initialization_std_for_query_key_rpe,
                conv_kernel_weight_decay=self.conv_kernel_weight_decay)

        if self.use_key_rpe_similarity:
            self.key_rpe = RelativePositionalEncoding(
                self.query_shape,
                self.memory_shape,
                self.key_depth_per_head,
                initialization_std=self.initialization_std_for_query_key_rpe,
                conv_kernel_weight_decay=self.conv_kernel_weight_decay)

        if self.retrieve_value_rpe:
            self.value_rpe = RelativePositionalEncoding(
                self.query_shape,
                self.memory_shape,
                self.total_value_depth // self.num_heads,
                initialization_std=self.initialization_std_for_value_rpe,
                conv_kernel_weight_decay=self.conv_kernel_weight_decay)


    def forward(self, input_tensor):
        batch_size, seq_len, input_channels = input_tensor.shape

        query_key_value = torch.einsum(
            'nlc,cd->nld', input_tensor, self.qkv_kernel)
        query_key_value = self.batch_norm_qkv(query_key_value)

        query, key, value = torch.split(
            query_key_value,
            [self.total_key_depth, self.total_key_depth, self.total_value_depth],
            dim=-1)

        query = query.view(batch_size, self.query_shape, self.num_heads,
                           self.key_depth_per_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads,
                       self.key_depth_per_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.total_value_depth // self.num_heads).transpose(1, 2)
        similarity_logits = []

        if self.use_content_similarity:
            content_similarity = torch.einsum(
                'bhld,bhmd->bhlm', query, key)
            similarity_logits.append(content_similarity)

        if self.use_query_rpe_similarity:
            # Implement query_rpe logic here
            query_rpe = self.query_rpe()
            query_rpe_similarity = torch.einsum(
                'bhld,lmd->bhlm', query, query_rpe)
            similarity_logits.append(query_rpe_similarity)


        if self.use_key_rpe_similarity:
            # Implement key_rpe logic here
            key_rpe = self.key_rpe()
            key_rpe_similarity = torch.einsum(
                'bhmd,lmd->bhlm', key, key_rpe)
            similarity_logits.append(key_rpe_similarity)
            

        similarity_logits = torch.stack(similarity_logits)
        similarity_logits = self.batch_norm_similarity(similarity_logits)
        similarity_logits = torch.sum(similarity_logits, dim=0)

        weights = self.attention_activate_fn(similarity_logits, dim=-1)

        retrieve_list = []

        if self.retrieve_value_content:
            retrieved_content = torch.einsum(
                'bhlm,bmhd->bhld', weights, value)
            retrieve_list.append(retrieved_content)

        if self.retrieve_value_rpe:
            # Implement value_rpe logic here
            value_rpe = self.value_rpe()
            retrieved_rpe = torch.einsum(
                'bhlm,lmd->bhld', weights, value_rpe)
            retrieve_list.append(retrieved_rpe)

        retrieved_output = torch.stack(retrieve_list)
        retrieved_output = self.batch_norm_retrieved_output(retrieved_output)
        retrieved_output = torch.sum(retrieved_output, dim=0)

        retrieved_output = retrieved_output.transpose(1, 2).contiguous()
        retrieved_output = retrieved_output.view(batch_size, seq_len, self.total_value_depth)

        return retrieved_output