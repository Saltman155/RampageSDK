// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor npu_add_custom(const at::Tensor& x, const at::Tensor& y)
{
    TORCH_CHECK_NPU(x);
    TORCH_CHECK_NPU(y);
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    TORCH_CHECK(x.dtype() == y.dtype(), "x and y must have the same dtype");

    at::Tensor output = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnAddCustom, x, y, output);
    return output;
}
