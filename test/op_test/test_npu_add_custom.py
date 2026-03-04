"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from rampage import npu_add_custom


def gen_inputs(input_shape, dtype):
    x = torch.randn(input_shape, dtype=dtype)
    y = torch.randn(input_shape, dtype=dtype)
    return x, y


def gen_cpu_outputs(x, y):
    return x + y


def gen_npu_outputs(x, y):
    npu_result = npu_add_custom(x.npu(), y.npu())
    return npu_result.cpu()


class TestNpuAddCustom(TestCase):
    def test_float16_small(self, device='npu'):
        x, y = gen_inputs([1024], torch.float16)
        cpu_result = gen_cpu_outputs(x, y)
        npu_result = gen_npu_outputs(x, y)
        self.assertRtolEqual(cpu_result.detach().numpy(), npu_result.detach().numpy())

    def test_float16_large(self, device='npu'):
        x, y = gen_inputs([8, 2048], torch.float16)
        cpu_result = gen_cpu_outputs(x, y)
        npu_result = gen_npu_outputs(x, y)
        self.assertRtolEqual(cpu_result.detach().numpy(), npu_result.detach().numpy())

    def test_float32_small(self, device='npu'):
        x, y = gen_inputs([1024], torch.float32)
        cpu_result = gen_cpu_outputs(x, y)
        npu_result = gen_npu_outputs(x, y)
        self.assertRtolEqual(cpu_result.detach().numpy(), npu_result.detach().numpy())

    def test_float32_large(self, device='npu'):
        x, y = gen_inputs([8, 2048], torch.float32)
        cpu_result = gen_cpu_outputs(x, y)
        npu_result = gen_npu_outputs(x, y)
        self.assertRtolEqual(cpu_result.detach().numpy(), npu_result.detach().numpy())

    def test_float32_2d(self, device='npu'):
        x, y = gen_inputs([256, 256], torch.float32)
        cpu_result = gen_cpu_outputs(x, y)
        npu_result = gen_npu_outputs(x, y)
        self.assertRtolEqual(cpu_result.detach().numpy(), npu_result.detach().numpy())

    def test_float16_3d(self, device='npu'):
        x, y = gen_inputs([4, 32, 64], torch.float16)
        cpu_result = gen_cpu_outputs(x, y)
        npu_result = gen_npu_outputs(x, y)
        self.assertRtolEqual(cpu_result.detach().numpy(), npu_result.detach().numpy())


if __name__ == "__main__":
    run_tests()
