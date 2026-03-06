"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Test suite for BEV Pool v1 operator on Ascend NPU.
This module tests the BEV pooling operation used in 3D object detection pipelines.
"""
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from rampage import npu_bev_pool_v1
import numpy as np


def gen_bev_inputs(n, c, b, d, h, w, dtype=torch.float32):
    """
    Generate test inputs for BEV pool operation.
    
    Args:
        n: Number of points/samples
        c: Channel dimension
        b: Batch size
        d: Depth dimension
        h: Height dimension
        w: Width dimension
        dtype: Data type
        
    Returns:
        feat: Feature tensor [N, C]
        geom_feat: Geometry indices tensor [N, 4] with values in range [0, d), [0, w), [0, b), [0, h)
    """
    feat = torch.randn(n, c, dtype=dtype)
    
    # Generate random geometry indices
    # Each row: [h_idx, w_idx, b_idx, d_idx]
    geom_feat = torch.zeros((n, 4), dtype=torch.int32)
    geom_feat[:, 0] = torch.randint(0, h, (n,), dtype=torch.int32)  # h_idx
    geom_feat[:, 1] = torch.randint(0, w, (n,), dtype=torch.int32)  # w_idx
    geom_feat[:, 2] = torch.randint(0, b, (n,), dtype=torch.int32)  # b_idx
    geom_feat[:, 3] = torch.randint(0, d, (n,), dtype=torch.int32)  # d_idx
    
    return feat, geom_feat


def gen_cpu_outputs(feat, geom_feat, b, d, h, w, c):
    """
    Generate CPU reference output using scatter add operation.
    
    Args:
        feat: Feature tensor [N, C]
        geom_feat: Geometry indices tensor [N, 4]
        b, d, h, w, c: Output dimensions
        
    Returns:
        output: Reference output tensor [B, C, D, H, W]
    """
    n = feat.shape[0]
    output = torch.zeros((b, c, d, h, w), dtype=feat.dtype)
    
    for i in range(n):
        h_idx = geom_feat[i, 0].item()
        w_idx = geom_feat[i, 1].item()
        b_idx = geom_feat[i, 2].item()
        d_idx = geom_feat[i, 3].item()
        
        # Accumulate features: output[b_idx, :, d_idx, h_idx, w_idx] += feat[i, :]
        output[b_idx, :, d_idx, h_idx, w_idx] += feat[i, :]
    
    return output


def gen_npu_outputs(feat, geom_feat, b, d, h, w, c):
    """
    Generate NPU output using BEV pool v1 operator.
    """
    npu_result = npu_bev_pool_v1(
        feat.npu(), 
        geom_feat.npu(), 
        b=b, d=d, h=h, w=w, c=c
    )
    return npu_result.cpu()


class TestNpuBevPoolV1(TestCase):
    """Test cases for BEV Pool v1 operator"""
    
    def test_small_float32(self, device='npu'):
        """Test with small tensors and float32"""
        n, c, b, d, h, w = 1000, 64, 2, 8, 64, 64
        feat, geom_feat = gen_bev_inputs(n, c, b, d, h, w, dtype=torch.float32)
        
        cpu_result = gen_cpu_outputs(feat, geom_feat, b, d, h, w, c)
        npu_result = gen_npu_outputs(feat, geom_feat, b, d, h, w, c)
        
        self.assertRtolEqual(
            cpu_result.detach().numpy(), 
            npu_result.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_medium_float32(self, device='npu'):
        """Test with medium-sized tensors and float32"""
        n, c, b, d, h, w = 50000, 128, 4, 16, 200, 200
        feat, geom_feat = gen_bev_inputs(n, c, b, d, h, w, dtype=torch.float32)
        
        cpu_result = gen_cpu_outputs(feat, geom_feat, b, d, h, w, c)
        npu_result = gen_npu_outputs(feat, geom_feat, b, d, h, w, c)
        
        self.assertRtolEqual(
            cpu_result.detach().numpy(), 
            npu_result.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_small_float16(self, device='npu'):
        """Test with small tensors and float16"""
        n, c, b, d, h, w = 1000, 64, 2, 8, 64, 64
        feat, geom_feat = gen_bev_inputs(n, c, b, d, h, w, dtype=torch.float16)
        
        cpu_result = gen_cpu_outputs(feat, geom_feat, b, d, h, w, c)
        npu_result = gen_npu_outputs(feat, geom_feat, b, d, h, w, c)
        
        self.assertRtolEqual(
            cpu_result.detach().to(torch.float32).numpy(), 
            npu_result.detach().to(torch.float32).numpy(),
            rtol=1e-3,
            atol=1e-3
        )

    def test_medium_float16(self, device='npu'):
        """Test with medium-sized tensors and float16"""
        n, c, b, d, h, w = 30000, 128, 4, 16, 200, 200
        feat, geom_feat = gen_bev_inputs(n, c, b, d, h, w, dtype=torch.float16)
        
        cpu_result = gen_cpu_outputs(feat, geom_feat, b, d, h, w, c)
        npu_result = gen_npu_outputs(feat, geom_feat, b, d, h, w, c)
        
        self.assertRtolEqual(
            cpu_result.detach().to(torch.float32).numpy(), 
            npu_result.detach().to(torch.float32).numpy(),
            rtol=1e-3,
            atol=1e-3
        )

    def test_various_dimensions(self, device='npu'):
        """Test with various output dimensions"""
        test_cases = [
            (5000, 64, 1, 8, 64, 64),
            (10000, 128, 2, 16, 128, 128),
            (20000, 256, 4, 20, 200, 200),
            (15000, 96, 3, 12, 150, 150),
        ]
        
        for n, c, b, d, h, w in test_cases:
            with self.subTest(n=n, c=c, b=b, d=d, h=h, w=w):
                feat, geom_feat = gen_bev_inputs(n, c, b, d, h, w, dtype=torch.float32)
                
                cpu_result = gen_cpu_outputs(feat, geom_feat, b, d, h, w, c)
                npu_result = gen_npu_outputs(feat, geom_feat, b, d, h, w, c)
                
                self.assertRtolEqual(
                    cpu_result.detach().numpy(), 
                    npu_result.detach().numpy(),
                    rtol=1e-5,
                    atol=1e-5
                )

    def test_output_shape(self, device='npu'):
        """Test that output shape is correct"""
        n, c, b, d, h, w = 5000, 128, 4, 16, 200, 200
        feat, geom_feat = gen_bev_inputs(n, c, b, d, h, w, dtype=torch.float32)
        
        output = npu_bev_pool_v1(
            feat.npu(), 
            geom_feat.npu(), 
            b=b, d=d, h=h, w=w, c=c
        )
        
        self.assertEqual(output.shape, torch.Size([b, c, d, h, w]))

    def test_data_accumulation(self, device='npu'):
        """Test that features are correctly accumulated to the same location"""
        n, c, b, d, h, w = 100, 32, 1, 4, 8, 8
        
        feat = torch.ones((n, c), dtype=torch.float32)
        geom_feat = torch.zeros((n, 4), dtype=torch.int32)
        
        # All points map to the same location
        geom_feat[:, :] = 0  # All points go to [0, 0, 0, 0]
        
        cpu_result = gen_cpu_outputs(feat, geom_feat, b, d, h, w, c)
        npu_result = gen_npu_outputs(feat, geom_feat, b, d, h, w, c)
        
        # Expected: output[0, :, 0, 0, 0] should be sum of all features = n
        expected_value = float(n)
        
        self.assertRtolEqual(
            cpu_result.detach().numpy(), 
            npu_result.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )
        
        # Verify the specific location value
        self.assertRtolEqual(
            cpu_result[0, 0, 0, 0, 0].item(),
            npu_result[0, 0, 0, 0, 0].item(),
            rtol=1e-5,
            atol=1e-5
        )


if __name__ == "__main__":
    run_tests()
