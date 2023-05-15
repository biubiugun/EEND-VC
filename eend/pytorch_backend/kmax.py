# A naive k-max used for speaker clustering written by Bohan Li 2023 SJTU

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from transformer import PositionalEncoding, PositionwiseFeedForward, MultiHeadSelfAttention

"""
Transformer Framework modified in k-means Mask Transformer. [1]

[1] k-means Mask Transformer, ECCV 2022.
      Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
      Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

"""

class KMaxTransformerDecoderLayer(nn.Module):
    r"""KMaxTransformerDecoderLayer is modified from torch.nn.TransformerDecoderLayer

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models.
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).

    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(KMaxTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.kmax_cross_attn = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, centers: Tensor, memory: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            centers: the sequence of k-means centers to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).

        """

        x = centers
        # x = self.linear_in(centers)

        x = self.norm1(x + self._kca_block(x, memory))
        x = self.norm2(x + self._sa_block(x))
        x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    # k-max cross attention block
    def _kca_block(self, x: Tensor, mem: Tensor) -> Tensor:
        x = self.kmax_cross_attn(x, mem, mem, need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_units,
                 e_units=2048, h=8, dropout_rate=0.1, layer_norm_eps: float = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.lnorm_in = nn.LayerNorm(n_units)
        self.pos_enc = PositionalEncoding(n_units, dropout_rate, 5000)
        self.dropout = nn.Dropout(p=dropout_rate)

        setattr(self, '_self_att',
                    MultiHeadSelfAttention(n_units, h, dropout_rate))
        setattr(self, '_lnorm',
                    nn.LayerNorm(n_units))
        setattr(self, '_ff',
                    PositionwiseFeedForward(n_units, e_units, dropout_rate))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):

        e = x

        # self-attention
        s = getattr(self, '_self_att')(e, x.shape[0] * x.shape[1]).view(x.shape[0], x.shape[1], -1)
        # residual
        e = e + self.dropout(s)
        # layer normalization
        e = getattr(self, '_lnorm')(e)
        # positionwise feed-forward
        s = getattr(self, '_ff')(e)
        # residual
        e = e + self.dropout(s)
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)

'''

Using traditional transformer block instead of axial-block(CV tasks) as the feature path in diarization task. (feature path)

Every output feature of a transformer encoder will be passed to two kmax-decoders for k-means center-updating. (centers path)

'''

class KMaxTransformer(nn.Module):
    '''
    Initial basic framework. Feasible to add the layer number as a parameter instead of a fixed layer number (3).
    '''
    
    def __init__(self, idim: int, num_mask_slots: int, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super(KMaxTransformer, self).__init__()
        self.init_centers = nn.Parameter( torch.randn(1, num_mask_slots, d_model) * 1.0)
        self.linear_in = nn.Linear(idim, d_model)
        self.lnorm = nn.LayerNorm(d_model)

        self.enc1 = TransformerEncoderLayer(d_model, dim_feedforward, nhead, dropout, layer_norm_eps)
        self.enc2 = TransformerEncoderLayer(d_model, dim_feedforward, nhead, dropout, layer_norm_eps)
        self.enc3 = TransformerEncoderLayer(d_model, dim_feedforward, nhead, dropout, layer_norm_eps)

        self.dec11 = KMaxTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.dec12 = KMaxTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.dec21 = KMaxTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.dec22 = KMaxTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.dec31 = KMaxTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.dec32 = KMaxTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)

    def _prepare_centers(self, batch_size):
        cluster_centers = self.init_centers.repeat(batch_size, 1, 1)
        return cluster_centers

    def forward(self, x):
        batch_size = x.shape[0]
        chunk_size = x.shape[1]
        cluster_centers = self._prepare_centers(batch_size)

        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = batch_size * chunk_size
        # e: (BT, F)
        x = self.lnorm(self.linear_in(x.reshape(BT_size, -1))).view(batch_size, chunk_size, -1)

        cluster_centers = self.dec12(self.dec11(cluster_centers, x), x)
        x = self.enc1(x)

        cluster_centers = self.dec22(self.dec21(cluster_centers, x), x)
        x = self.enc2(x)

        cluster_centers = self.dec32(self.dec31(cluster_centers, x), x)
        x = self.enc3(x)

        # (B x T x F) x (B x N x F)
        kmax_result = torch.einsum('btf,bnf->btn', x, cluster_centers)
        
        return kmax_result, cluster_centers



