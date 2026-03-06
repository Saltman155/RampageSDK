"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

BEV Pool v1 Operator - Bird's Eye View pooling for autonomous driving perception.
This module provides PyTorch interface for the BEV pooling operation on Ascend NPU.
"""
from torch.autograd import Function
import torch
import rampage._C


class BEVPoolV1Function(Function):
    @staticmethod
    def forward(ctx, feat, geom_feat, b, d, h, w, c):
        """
        Forward pass for BEV pool v1 operation.
        
        Args:
            feat: Input features tensor of shape [N, C]
            geom_feat: Geometry indices tensor of shape [N, 4] with values [h_idx, w_idx, b_idx, d_idx]
            b: Batch size
            d: Depth dimension
            h: Height dimension
            w: Width dimension
            c: Channel dimension
            
        Returns:
            Output tensor of shape [B, C, D, H, W]
        """
        output = rampage._C.npu_bev_pool_v1(feat, geom_feat, b, d, h, w, c)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # BEV pool is a one-way operation, backward is not implemented yet
        raise NotImplementedError("Backward for BEV pool v1 is not implemented yet")


def npu_bev_pool_v1(feat, geom_feat, b, d, h, w, c):
    """
    BEV (Bird's Eye View) pooling operation for autonomous driving perception.
    
    This operation accumulates features from a sparse 3D space into a dense 
    bird's eye view representation, commonly used in 3D object detection pipelines.
    
    Args:
        feat: Input features tensor of shape [N, C], dtype: float32/float16
        geom_feat: Geometry indices tensor of shape [N, 4], dtype: int32
                   Each row contains [h_idx, w_idx, b_idx, d_idx] indicating
                   which position in the output tensor to accumulate to
        b: Batch size (int)
        d: Depth dimension size (int)
        h: Height dimension size (int)
        w: Width dimension size (int)
        c: Channel dimension size (int)
        
    Returns:
        Output tensor of shape [B, C, D, H, W] with accumulated features
        
    Example:
        >>> feat = torch.randn(10000, 256, dtype=torch.float32).npu()
        >>> geom_feat = torch.randint(0, 100, (10000, 4), dtype=torch.int32).npu()
        >>> output = npu_bev_pool_v1(feat, geom_feat, b=4, d=16, h=200, w=200, c=256)
        >>> output.shape
        torch.Size([4, 256, 16, 200, 200])
    """
    return BEVPoolV1Function.apply(feat, geom_feat, b, d, h, w, c)
