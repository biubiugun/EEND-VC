# A naive k-max used for speaker clustering written by Bohan Li 2023 SJTU

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from transformer import PositionalEncoding, PositionwiseFeedForward, MultiHeadSelfAttention

"""
Transformer Decoder used in k-means Mask Transformer. [1]

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
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
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

        Shape:
            see the docs in Transformer class.
        """

        x = centers
        
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



class KMaxTransforemer(nn.Module):
    
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super(KMaxTransforemer, self).__init__()
        self.centers = None