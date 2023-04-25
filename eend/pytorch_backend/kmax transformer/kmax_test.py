import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from transformer import PositionalEncoding, PositionwiseFeedForward, MultiHeadSelfAttention
from kmax import KMaxTransformer

cluster_num = 7
chunk_size = 300
freq = 512
batch_size = 16

model = KMaxTransformer(512, 1024, 512, 1)
model.initialize_cluster_centers(batch_size, 3, freq)


def test():
    x = torch.rand((batch_size, chunk_size, freq))
    print(x.shape)
    result = model(x)
    print(result.shape)

if __name__ == '__main__':
    test()