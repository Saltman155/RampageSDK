// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
namespace {
int32_t SMALLEST_TENSOR_SIZE = 1;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_unique(
    const at::Tensor& input, bool return_inverse, bool return_counts)
{
    TORCH_CHECK_NPU(input);
    if (input.numel() <= SMALLEST_TENSOR_SIZE) {
        at::Tensor output = at::Tensor(input).clone();
        at::Tensor uniqueCnt = at::tensor({input.numel()}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device()));
        at::Tensor inverse = return_inverse ?
            at::zeros({input.numel()}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device())) :
            at::empty({0}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device()));
        at::Tensor counts = return_counts ?
            at::tensor({input.numel()}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device())) :
            at::empty({0}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device()));
        return std::make_tuple(output, uniqueCnt, inverse, counts);
    } else {
        at::Tensor output = at::empty({input.numel()}, at::TensorOptions().dtype(input.dtype()).device(input.device()));
        at::Tensor uniqueCnt = at::empty({1}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device()));
        at::Tensor inverse = at::empty({input.numel()}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device()));
        at::Tensor counts = at::empty({input.numel()}, at::TensorOptions().dtype(at::ScalarType::Int).device(input.device()));
        EXEC_NPU_CMD_SYNC(aclnnUniqueV3, input, output, uniqueCnt, inverse, counts, return_inverse, return_counts);
        int uniqueCount = uniqueCnt.item<int>();
        output = output.narrow(0, 0, uniqueCount);
        if (return_counts) {
            counts = counts.narrow(0, 0, uniqueCount);
        }
        return std::make_tuple(output, uniqueCnt, inverse, counts);
    }
}
