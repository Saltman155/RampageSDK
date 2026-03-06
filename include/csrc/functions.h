// Copyright (c) 2024, Huawei Technologies.All rights reserved.
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
#ifndef CSRC_FUNCTIONS_H_
#define CSRC_FUNCTIONS_H_

#include <ATen/ATen.h>

//请在此处添加算子接口声明

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_unique(
    const at::Tensor& input, bool return_inverse, bool return_counts);

at::Tensor npu_add_custom(const at::Tensor& x, const at::Tensor& y);

#endif // CSRC_FUNCTIONS_H_
