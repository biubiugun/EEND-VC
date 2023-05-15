import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from kmax import KMaxTransformer


def test_kmax_transformer():
    idim = 345
    num_mask_slots = 3
    d_model = 512
    batch_size = 1024
    sequence_length = 150

    # 创建输入张量，尺寸为 (batch_size, sequence_length, idim)
    input_tensor = torch.randn(batch_size, sequence_length, idim)

    # 创建 KMaxTransformer 对象
    kmax_transformer = KMaxTransformer(idim, num_mask_slots, d_model)

    # 运行前向传递并获取输出
    kmax_result, cluster_centers = kmax_transformer(input_tensor)

    # 检查输出尺寸
    assert kmax_result.shape == (batch_size, sequence_length, num_mask_slots), f"Expected output shape: {(batch_size, sequence_length, num_mask_slots)}, got: {kmax_result.shape}"
    assert cluster_centers.shape == (batch_size, num_mask_slots, d_model), f"Expected cluster centers shape: {(batch_size, num_mask_slots, d_model)}, got: {cluster_centers.shape}"

    print("KMaxTransformer 测试通过！")

# 运行测试函数
test_kmax_transformer()
