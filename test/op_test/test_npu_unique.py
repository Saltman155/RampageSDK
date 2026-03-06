"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import random
import os
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from rampage import npu_unique


def gen_inputs(input_shape, dtype):
    input_tensor = torch.randint(-256, 256, input_shape, dtype=dtype)
    return input_tensor


def gen_cpu_outputs(input_tensor, return_inverse=False, return_counts=False):
    cpu_result = torch.unique(input_tensor, sorted=True,
                              return_inverse=return_inverse,
                              return_counts=return_counts)
    return cpu_result


def gen_npu_outputs(input_tensor, return_inverse=False, return_counts=False):
    npu_result = npu_unique(input_tensor.npu(), return_inverse, return_counts)
    if isinstance(npu_result, tuple):
        return tuple(r.cpu() if hasattr(r, 'cpu') else r for r in npu_result)
    return npu_result.cpu()


class TestNpuUnique(TestCase):
    # ========== Basic unique tests (no inverse/counts) ==========
    def test_bfloat16(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.bfloat16)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().to(torch.float32).numpy(),
                             npu_result.cpu().detach().to(torch.float32).numpy())

    def test_float16(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.float16)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_float32(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.float32)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_int16(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.int16)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_int32(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.int32)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_int64(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.int64)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    # ========== Tests with return_inverse ==========
    def test_float32_inverse(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 100000)], torch.float32)
        cpu_output, cpu_inverse = gen_cpu_outputs(input_tensor, return_inverse=True)
        npu_output, npu_unique_cnt, npu_inverse = gen_npu_outputs(input_tensor, return_inverse=True)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        self.assertRtolEqual(cpu_inverse.numpy(), npu_inverse[:input_tensor.numel()].numpy())

    def test_int32_inverse(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 100000)], torch.int32)
        cpu_output, cpu_inverse = gen_cpu_outputs(input_tensor, return_inverse=True)
        npu_output, npu_unique_cnt, npu_inverse = gen_npu_outputs(input_tensor, return_inverse=True)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        self.assertRtolEqual(cpu_inverse.numpy(), npu_inverse[:input_tensor.numel()].numpy())

    # ========== Tests with return_counts ==========
    def test_float32_counts(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 100000)], torch.float32)
        cpu_output, cpu_counts = gen_cpu_outputs(input_tensor, return_counts=True)
        npu_output, npu_unique_cnt, npu_counts = gen_npu_outputs(input_tensor, return_counts=True)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        self.assertRtolEqual(cpu_counts.numpy(), npu_counts.numpy())

    def test_int32_counts(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 100000)], torch.int32)
        cpu_output, cpu_counts = gen_cpu_outputs(input_tensor, return_counts=True)
        npu_output, npu_unique_cnt, npu_counts = gen_npu_outputs(input_tensor, return_counts=True)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        self.assertRtolEqual(cpu_counts.numpy(), npu_counts.numpy())

    # ========== Tests with both inverse and counts ==========
    def test_float32_inverse_counts(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 100000)], torch.float32)
        cpu_output, cpu_inverse, cpu_counts = gen_cpu_outputs(
            input_tensor, return_inverse=True, return_counts=True)
        npu_output, npu_unique_cnt, npu_inverse, npu_counts = gen_npu_outputs(
            input_tensor, return_inverse=True, return_counts=True)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        self.assertRtolEqual(cpu_inverse.numpy(), npu_inverse[:input_tensor.numel()].numpy())
        self.assertRtolEqual(cpu_counts.numpy(), npu_counts.numpy())

    def test_int32_inverse_counts(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 100000)], torch.int32)
        cpu_output, cpu_inverse, cpu_counts = gen_cpu_outputs(
            input_tensor, return_inverse=True, return_counts=True)
        npu_output, npu_unique_cnt, npu_inverse, npu_counts = gen_npu_outputs(
            input_tensor, return_inverse=True, return_counts=True)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        self.assertRtolEqual(cpu_inverse.numpy(), npu_inverse[:input_tensor.numel()].numpy())
        self.assertRtolEqual(cpu_counts.numpy(), npu_counts.numpy())


if __name__ == "__main__":
    run_tests()
