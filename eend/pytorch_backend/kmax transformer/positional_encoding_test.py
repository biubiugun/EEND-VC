import torch
import torch.nn as nn
import numpy as np

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
        self._initializer = torch.nn.init.trunc_normal_
        self._regularizer = nn.functional.normalize

        self._relative_distance_matrix = _compute_relative_distance_matrix(query_length, key_length)
        self._embedding_shape = (MAX_SPAN * 2 - 1, depth)
        self.conv_kernel_weight_decay = conv_kernel_weight_decay

        self._embeddings = torch.Tensor(*self._embedding_shape)
        self.reset_parameters()

    def reset_parameters(self):
        self._initializer(self._embeddings)
        

    def forward(self, inputs):
        """A forward pass that gathers the relative positional encoding."""
        del inputs
        # Gather the embeddings according to the relative distances.
        return torch.index_select(self._embeddings, 0, self._relative_distance_matrix.view(-1)).view(self._relative_distance_matrix.shape[0], self._relative_distance_matrix.shape[1], self._embeddings.shape[1])

rpe = RelativePositionalEncoding(10, 10, 10, 'rpe')
print(rpe)