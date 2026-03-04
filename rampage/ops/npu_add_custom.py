"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
from torch.autograd import Function
import rampage._C


class AddCustomFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        z = rampage._C.npu_add_custom(x, y)
        return z


npu_add_custom = AddCustomFunction.apply
