"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
Modification 2. Upgrade to UniqueV3 with inverse and counts support
"""
from torch.autograd import Function
import rampage._C


class UniqueFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, input_tensor, return_inverse=False, return_counts=False):
        output, unique_cnt, inverse, counts = rampage._C.npu_unique(
            input_tensor, return_inverse, return_counts)
        if return_inverse and return_counts:
            return output, unique_cnt, inverse, counts
        elif return_inverse:
            return output, unique_cnt, inverse
        elif return_counts:
            return output, unique_cnt, counts
        else:
            return output


npu_unique = UniqueFunction.apply
