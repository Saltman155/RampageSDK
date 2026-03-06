#!/usr/bin/env python3
"""
BEV Pool v1 Quick Test Script
直接在PyTorch中测试BEV Pool v1算子的独立脚本
"""

import torch
import torch_npu
import numpy as np
from rampage import npu_bev_pool_v1


def cpu_bev_pool_reference(feat, geom_feat, b, d, h, w, c):
    """
    CPU参考实现：使用PyTorch的scatter操作
    """
    n = feat.shape[0]
    output = torch.zeros((b, c, d, h, w), dtype=feat.dtype)
    
    for i in range(n):
        h_idx = geom_feat[i, 0].item()
        w_idx = geom_feat[i, 1].item()
        b_idx = geom_feat[i, 2].item()
        d_idx = geom_feat[i, 3].item()
        
        # 边界检查
        if (0 <= h_idx < h and 0 <= w_idx < w and 
            0 <= b_idx < b and 0 <= d_idx < d):
            output[b_idx, :, d_idx, h_idx, w_idx] += feat[i, :]
    
    return output


def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("测试1: 基础功能 (Float32, 小规模)")
    print("=" * 60)
    
    n, c, b, d, h, w = 1000, 64, 2, 8, 64, 64
    
    # 生成输入
    feat = torch.randn(n, c, dtype=torch.float32)
    geom_feat = torch.zeros((n, 4), dtype=torch.int32)
    geom_feat[:, 0] = torch.randint(0, h, (n,))  # h_idx
    geom_feat[:, 1] = torch.randint(0, w, (n,))  # w_idx
    geom_feat[:, 2] = torch.randint(0, b, (n,))  # b_idx
    geom_feat[:, 3] = torch.randint(0, d, (n,))  # d_idx
    
    print(f"输入feat: {feat.shape}, dtype: {feat.dtype}")
    print(f"输入geom_feat: {geom_feat.shape}, dtype: {geom_feat.dtype}")
    
    # CPU参考
    cpu_output = cpu_bev_pool_reference(feat, geom_feat, b, d, h, w, c)
    print(f"CPU参考输出: {cpu_output.shape}")
    print(f"CPU输出范围: [{cpu_output.min():.4f}, {cpu_output.max():.4f}]")
    
    # NPU计算
    try:
        npu_output = npu_bev_pool_v1(feat.npu(), geom_feat.npu(), b, d, h, w, c)
        npu_output_cpu = npu_output.cpu()
        print(f"NPU输出: {npu_output_cpu.shape}")
        print(f"NPU输出范围: [{npu_output_cpu.min():.4f}, {npu_output_cpu.max():.4f}]")
        
        # 比较
        diff = torch.abs(cpu_output - npu_output_cpu)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"差值统计:")
        print(f"  最大差值: {max_diff:.2e}")
        print(f"  平均差值: {mean_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✓ 测试PASSED - 结果一致")
            return True
        else:
            print("✗ 测试FAILED - 结果差异过大")
            return False
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        return False


def test_float16():
    """Float16精度测试"""
    print("\n" + "=" * 60)
    print("测试2: Float16精度")
    print("=" * 60)
    
    n, c, b, d, h, w = 500, 32, 1, 4, 32, 32
    
    feat = torch.randn(n, c, dtype=torch.float16)
    geom_feat = torch.zeros((n, 4), dtype=torch.int32)
    geom_feat[:, 0] = torch.randint(0, h, (n,))
    geom_feat[:, 1] = torch.randint(0, w, (n,))
    geom_feat[:, 2] = torch.randint(0, b, (n,))
    geom_feat[:, 3] = torch.randint(0, d, (n,))
    
    print(f"输入feat: {feat.shape}, dtype: {feat.dtype}")
    
    cpu_output = cpu_bev_pool_reference(feat, geom_feat, b, d, h, w, c)
    
    try:
        npu_output = npu_bev_pool_v1(feat.npu(), geom_feat.npu(), b, d, h, w, c)
        npu_output_cpu = npu_output.cpu()
        
        # float16对比时转换为float32
        cpu_fp32 = cpu_output.to(torch.float32)
        npu_fp32 = npu_output_cpu.to(torch.float32)
        
        diff = torch.abs(cpu_fp32 - npu_fp32)
        max_diff = diff.max().item()
        
        print(f"差值 (float32比较): {max_diff:.2e}")
        
        if max_diff < 1e-3:
            print("✓ 测试PASSED")
            return True
        else:
            print("✗ 测试FAILED")
            return False
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        return False


def test_accumulation():
    """特征累加测试"""
    print("\n" + "=" * 60)
    print("测试3: 特征累加(多个点映射到同一位置)")
    print("=" * 60)
    
    n, c, b, d, h, w = 100, 32, 1, 4, 8, 8
    
    # 所有点的特征都是1，映射到同一个位置
    feat = torch.ones((n, c), dtype=torch.float32)
    geom_feat = torch.zeros((n, 4), dtype=torch.int32)  # 所有都映射到[0, 0, 0, 0]
    
    print(f"设置: {n}个点，特征值均为1.0，全部映射到位置[0, 0, 0, 0]")
    print(f"期望该位置的累加值为: {n}.0")
    
    cpu_output = cpu_bev_pool_reference(feat, geom_feat, b, d, h, w, c)
    expected_value = cpu_output[0, 0, 0, 0, 0].item()
    
    try:
        npu_output = npu_bev_pool_v1(feat.npu(), geom_feat.npu(), b, d, h, w, c)
        npu_output_cpu = npu_output.cpu()
        
        actual_value = npu_output_cpu[0, 0, 0, 0, 0].item()
        
        print(f"CPU计算值: {expected_value:.1f}")
        print(f"NPU计算值: {actual_value:.1f}")
        
        if abs(expected_value - actual_value) < 0.1:
            print("✓ 测试PASSED - 累加正确")
            return True
        else:
            print("✗ 测试FAILED - 累加错误")
            return False
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        return False


def test_output_shape():
    """输出形状测试"""
    print("\n" + "=" * 60)
    print("测试4: 输出形状验证")
    print("=" * 60)
    
    test_cases = [
        (1000, 64, 1, 8, 64, 64),
        (5000, 128, 2, 16, 100, 100),
        (10000, 256, 4, 20, 200, 200),
    ]
    
    passed = 0
    for n, c, b, d, h, w in test_cases:
        feat = torch.randn(n, c, dtype=torch.float32)
        geom_feat = torch.zeros((n, 4), dtype=torch.int32)
        geom_feat[:, 0] = torch.randint(0, h, (n,))
        geom_feat[:, 1] = torch.randint(0, w, (n,))
        geom_feat[:, 2] = torch.randint(0, b, (n,))
        geom_feat[:, 3] = torch.randint(0, d, (n,))
        
        try:
            output = npu_bev_pool_v1(feat.npu(), geom_feat.npu(), b, d, h, w, c)
            expected_shape = (b, c, d, h, w)
            actual_shape = tuple(output.shape)
            
            if expected_shape == actual_shape:
                print(f"  ✓ n={n}, c={c}, b={b}, d={d}, h={h}, w={w} -> {actual_shape}")
                passed += 1
            else:
                print(f"  ✗ n={n}, c={c}: 期望{expected_shape}, 得到{actual_shape}")
        except Exception as e:
            print(f"  ✗ n={n}, c={c}: 执行失败 {e}")
    
    print(f"形状验证: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_large_scale():
    """大规模测试"""
    print("\n" + "=" * 60)
    print("测试5: 大规模数据 (生产级别)")
    print("=" * 60)
    
    # 模拟真实场景：100K个点，256维特征，4批次
    n, c, b, d, h, w = 100000, 256, 4, 16, 200, 200
    
    print(f"数据规模:")
    print(f"  点数 N={n}")
    print(f"  通道数 C={c}")
    print(f"  批次/深度/高/宽: B={b}, D={d}, H={h}, W={w}")
    print(f"  输入大小: {n * c * 4 / 1e9:.2f} GB (feat)")
    print(f"  输出大小: {b * c * d * h * w * 4 / 1e9:.2f} GB")
    
    feat = torch.randn(n, c, dtype=torch.float32)
    geom_feat = torch.zeros((n, 4), dtype=torch.int32)
    geom_feat[:, 0] = torch.randint(0, h, (n,))
    geom_feat[:, 1] = torch.randint(0, w, (n,))
    geom_feat[:, 2] = torch.randint(0, b, (n,))
    geom_feat[:, 3] = torch.randint(0, d, (n,))
    
    try:
        import time
        
        # 预热
        _ = npu_bev_pool_v1(feat.npu(), geom_feat.npu(), b, d, h, w, c)
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        output = npu_bev_pool_v1(feat.npu(), geom_feat.npu(), b, d, h, w, c)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        input_bytes = n * c * 4
        output_bytes = b * c * d * h * w * 4
        total_bytes = (input_bytes * 2 + output_bytes) / 1e9  # feat + geom + output
        
        throughput = total_bytes / elapsed
        
        print(f"执行时间: {elapsed*1000:.2f} ms")
        print(f"吞吐量: {throughput:.2f} GB/s")
        print(f"✓ 大规模测试通过")
        return True
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "BEV Pool v1 测试套件" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = []
    
    # 运行测试
    results.append(("基础功能", test_basic()))
    results.append(("Float16精度", test_float16()))
    results.append(("特征累加", test_accumulation()))
    results.append(("输出形状", test_output_shape()))
    results.append(("大规模数据", test_large_scale()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20} {status}")
    
    print(f"\n总体: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！BEV Pool v1算子运行正常。")
        return 0
    else:
        print(f"\n⚠️  有{total - passed}个测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit(main())
