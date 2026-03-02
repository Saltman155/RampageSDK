# RampageSDK 算子库技术文档

## 1. 项目概述

RampageSDK 是一个面向华为昇腾（Ascend）NPU 的自定义算子库，旨在为自动驾驶系统提供加速支持。它基于 CANN（Compute Architecture for Neural Networks）工具链，通过 AscendC 编程模型编写高性能自定义算子，并通过 PyTorch + torch_npu 接口将算子暴露给 Python 用户使用。

### 1.1 核心特性

- 支持多种昇腾芯片架构：ascend910b、ascend910、ascend310p、ascend910_93、ascend910_95（arch35）
- 基于 AscendC 编程模型的高性能 Kernel 实现
- 与 PyTorch 深度集成（通过 pybind11 和 torch_npu）
- 三阶段 CMake 构建系统，自动完成算子注册、Tiling、二进制编译等全流程
- 支持动态形状（Dynamic Shape）算子

---

## 2. 项目目录结构

```
RampageSDK/
├── setup.py                    # Python 包构建入口（pip install 入口）
├── CMakeLists.txt              # 顶层 CMake 构建脚本
├── CMakePresets.json           # CMake 预设配置（芯片型号、路径等）
├── cmake/                      # CMake 工具函数和配置
│   ├── config.cmake            # 全局配置（CANN 路径、编译器路径、计算单元等）
│   ├── func.cmake              # CMake 自定义函数（opbuild、二进制编译、安装等）
│   ├── intf.cmake              # 编译选项接口库（编译 flags、链接选项等）
│   ├── makeself.cmake          # 打包相关
│   └── util/                   # Python/Shell 辅助脚本（代码生成、配置解析等）
├── kernels/                    # 标准架构算子 Kernel 源码
│   ├── CMakeLists.txt          # Kernel 构建逻辑（Stage 0 + Stage 1）
│   ├── op_host/                # Host 侧代码（算子定义、Tiling 函数、形状推导）
│   │   ├── common.h            # 通用工具函数
│   │   ├── unique_v2_tiling.h  # UniqueV2 算子 Tiling 数据定义
│   │   └── unique_v2.cpp       # UniqueV2 算子定义（InferShape + Tiling + OpDef）
│   └── op_kernel/              # Device 侧代码（AscendC Kernel 实现）
│       ├── unique_v2.h         # UniqueV2 Kernel 类实现
│       └── unique_v2.cpp       # UniqueV2 Kernel 入口函数
├── kernels_arch35/             # Arch35 架构（ascend910_95）专用 Kernel（结构同 kernels/）
│   └── CMakeLists.txt          # 当前为空，待添加 arch35 专属算子
├── rampage/                    # Python 包源码
│   ├── __init__.py             # 包初始化、环境配置、算子导出
│   ├── get_chip_info.py        # 芯片型号检测（区分 arch35）
│   ├── csrc/                   # C++ 扩展源码
│   │   ├── CMakeLists.txt      # C++ 扩展构建（Stage 2，生成 _C.so）
│   │   ├── pybind.cpp          # PyBind11 绑定入口
│   │   ├── Unique.cpp          # npu_unique 算子的 C++ 封装
│   │   └── OpApiCommon.cpp     # OpAPI 通用工具（参数序列化、哈希计算）
│   └── ops/                    # Python 算子接口
│       ├── __init__.py
│       └── npu_unique.py       # npu_unique 算子的 Python 封装
├── include/                    # 公共头文件
│   ├── csrc/
│   │   ├── functions.h         # 所有算子的 C++ 函数声明
│   │   ├── OpApiCommon.h       # OpAPI 调用宏（EXEC_NPU_CMD 等）
│   │   ├── common.h            # 通用定义
│   │   └── utils.h             # 工具函数
│   ├── ge/                     # GE（Graph Engine）相关头文件
│   ├── log/                    # 日志头文件
│   └── onnx/                   # ONNX 相关头文件
└── scripts/                    # 构建与安装脚本
    ├── build_kernel.sh         # 独立 Kernel 构建脚本
    ├── install_kernel.sh       # Kernel 安装到 OPP 路径的脚本
    ├── retry.sh                # 重试机制脚本
    └── upgrade_kernel.sh       # Kernel 升级脚本
```

---

## 3. 编译流程详解

### 3.1 编译入口：`setup.py`

RampageSDK 通过标准的 Python `setuptools` 进行构建和安装。执行 `pip install -e .`（开发模式）或 `pip install .`（正式安装）即可触发完整构建。

`setup.py` 定义了三个核心的自定义构建命令类：

| 命令类 | 构建阶段 | 职责 |
|--------|---------|------|
| `CPPLibBuild` (build_clib) | Stage 0 + Stage 1 | 构建算子 Host 库（proto、tiling、opapi）和 Kernel 二进制 |
| `ExtBuild` (build_ext) | Stage 2 | 构建 PyTorch C++ 扩展 `_C.so` |
| `DevelopBuild` (develop) | 全部 | 开发模式安装，依次调用 build_clib 和 build_ext |

#### 3.1.1 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `USE_ARCH35` | 是否使用 arch35 架构（ascend910_95） | `false` |
| `BUILD_WITHOUT_SHA` | 版本号是否不附加 git commit SHA | 未设置则附加 |

#### 3.1.2 构建命令

```bash
# 开发模式安装（推荐开发时使用）
python setup.py develop

# 正式安装
python setup.py bdist_wheel

# 只构建单个算子 Kernel（开发模式）
python setup.py develop --kernel-name="DeformableConv2d;MultiScaleDeformableAttn"

# 使用 arch35 架构
USE_ARCH35=true python setup.py xxxxxxx


### 3.2 三阶段构建详解

整个构建过程通过 CMake 变量 `BUILD_STAGE` 控制，分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                    完整构建流程                                   │
├─────────┬───────────────────────────────────────────────────────┤
│ Stage 0 │ 编译 op_host → 运行 opbuild → 生成自动代码            │
├─────────┼───────────────────────────────────────────────────────┤
│ Stage 1 │ 编译 proto/tiling/opapi 库 → 生成 ops info            │
│         │ → 生成 impl Python → 二进制编译                       │
├─────────┼───────────────────────────────────────────────────────┤
│ Stage 2 │ 编译 PyTorch C++ 扩展 _C.so                           │
└─────────┴───────────────────────────────────────────────────────┘
```

#### Stage 0：算子信息提取与代码生成

**入口**: `kernels/CMakeLists.txt` 中 `BUILD_STAGE EQUAL 0` 分支

**流程**：
1. 将 `kernels/op_host/*.cpp` 编译为共享库 `libascend_all_ops.so`
2. 调用 CANN 工具链的 `opbuild` 工具解析该共享库
3. 自动生成以下文件到 `${ASCEND_AUTOGEN_PATH}`（即 `build/autogen`）：
   - `op_proto.cc` / `op_proto.h` — 算子原型定义
   - `aclnn_*.cpp` / `aclnn_*.h` — ACLNN API 封装代码
   - `aic-<compute_unit>-ops-info.ini` — 各计算单元的算子信息文件

**关键 CMake 代码**：
```cmake
add_library(ascend_all_ops SHARED ${ASCEND_HOST_SRC})
# ... 编译选项 ...
add_custom_command(
    TARGET ascend_all_ops POST_BUILD
    COMMAND ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
            $<TARGET_FILE:ascend_all_ops> ${ASCEND_AUTOGEN_PATH})
```

#### Stage 1：算子库编译与二进制打包

**入口**: `kernels/CMakeLists.txt` 中 `BUILD_STAGE EQUAL 1` 分支

**编译产物**：

| 产物 | 说明 | 安装路径 |
|------|------|---------|
| `libcust_opsproto_rt2.0.so` | 算子原型库 | `packages/vendors/customize/op_proto/lib/linux/<arch>/` |
| `libcust_opmaster_rt2.0.so` | 算子 Tiling 库（含 Host 逻辑） | `packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/<arch>/` |
| `libcust_opapi.so` | ACLNN API 封装库 | `packages/vendors/customize/op_api/lib/` |
| `npu_supported_ops.json` | 支持的算子列表 | `packages/vendors/customize/framework/tensorflow/` |
| Kernel 二进制 | 各计算单元的预编译 Kernel | `packages/vendors/customize/op_impl/ai_core/tbe/kernel/<compute_unit>/` |
| `version.info` | 版本信息 | `packages/vendors/customize/` |

**详细流程**：

1. **编译 Proto 库**：将 `autogen/op_proto.cc` 编译为 `libcust_opsproto_rt2.0.so`
2. **编译 Tiling 库**：将 `op_host/*.cpp` 编译为 `libcust_opmaster_rt2.0.so`，创建 `liboptiling.so` 软链接
3. **编译 OpAPI 库**：将 `autogen/aclnn_*.cpp` 编译为 `libcust_opapi.so`
4. **遍历各计算单元**（ascend910b, ascend910, ascend310p, ascend910_93）：
   - 生成 `aic-<compute_unit>-ops-info.json`（调用 `parse_ini_to_json.py`）
   - 生成 AscendC impl Python 文件（调用 `ascendc_impl_build.py`）
   - 进行 Kernel 二进制编译（调用 `ascendc_bin_param_build.py` 生成编译脚本，逐个执行）
5. **生成 npu_supported_ops.json** 和 **version.info**

#### Stage 2：PyTorch C++ 扩展编译

**入口**: `rampage/csrc/CMakeLists.txt` 中 `BUILD_STAGE EQUAL 2` 分支

**流程**：
1. 查找 Python3、PyTorch、torch_npu 的路径
2. 编译 `rampage/csrc/*.cpp` 为共享库 `_C.<python_soabi>.so`
3. 输出到 `rampage/` 目录，供 Python 直接 `import rampage._C` 使用

**链接依赖**：
- `c10`、`torch`、`torch_python` — PyTorch 核心库
- `torch_npu` — 昇腾 NPU 的 PyTorch 后端

**关键编译选项**：
- C++17 标准
- PyBind11 ABI 兼容性标志
- `-DTORCH_EXTENSION_NAME=_C`

### 3.3 CMake 配置体系

#### 3.3.1 `CMakePresets.json` — 预设配置

```json
{
    "configurePresets": [{
        "name": "default",
        "generator": "Unix Makefiles",
        "cacheVariables": {
            "ASCEND_COMPUTE_UNIT": "ascend910b;ascend910;ascend310p;ascend910_93",
            "ASCEND_COMPUTE_UNIT_ARCH35": "ascend910_95",
            "ASCEND_CANN_PACKAGE_PATH": "/usr/local/Ascend/latest",
            "vendor_name": "customize",
            "vendor_name_arch35": "customize_arch35",
            "ENABLE_BINARY_PACKAGE": true
        }
    }]
}
```

| 变量 | 说明 |
|------|------|
| `ASCEND_COMPUTE_UNIT` | 标准架构支持的计算单元列表 |
| `ASCEND_COMPUTE_UNIT_ARCH35` | Arch35 架构支持的计算单元 |
| `ASCEND_CANN_PACKAGE_PATH` | CANN 工具链安装路径 |
| `vendor_name` | 算子包的厂商名称 |
| `ENABLE_BINARY_PACKAGE` | 是否进行 Kernel 二进制预编译 |
| `ENABLE_ONNX` | 是否编译 ONNX 插件 |

#### 3.3.2 `cmake/config.cmake` — 路径和环境检测

- 从环境变量 `ASCEND_AICPU_PATH` 或默认路径 `/usr/local/Ascend/latest` 检测 CANN 安装路径
- 通过 `ASCEND_CANN_PACKAGE_PATH` 的软链接定位编译器路径
- 检测系统架构（aarch64/x86_64）
- 设置自动代码生成目录 `ASCEND_AUTOGEN_PATH`

#### 3.3.3 `cmake/intf.cmake` — 编译接口库

定义了一个 INTERFACE 库 `intf_pub`，统一管理编译选项：

- **编译选项**: `-fPIC`、`-fvisibility=hidden`、C++11 标准
- **安全选项**: `-fstack-protector-strong`、`-Wl,-z,relro`、`-Wl,-z,now`
- **Release**: `-O2` + `_FORTIFY_SOURCE=2`
- **Debug**: `-O0 -g -ftrapv -fstack-check`
- **ABI**: `_GLIBCXX_USE_CXX11_ABI=0`

#### 3.3.4 `cmake/func.cmake` — 构建函数库

| 函数 | 功能 |
|------|------|
| `install_target()` | 安装编译目标到 RAMPAGE_PATH |
| `install_file()` | 安装文件到 RAMPAGE_PATH |
| `opbuild()` | 调用 CANN opbuild 工具生成算子代码 |
| `add_ops_info_target()` | 生成 ops-info.json（INI → JSON 转换） |
| `add_ops_impl_target()` | 生成 AscendC 实现的 Python 封装 |
| `add_npu_support_target()` | 生成 npu_supported_ops.json |
| `add_bin_compile_target()` | Kernel 二进制编译（动态形状） |
| `add_ops_compile_options()` | 添加自定义编译选项 |

### 3.4 独立 Kernel 构建

除了通过 `setup.py` 构建外，也可以使用独立脚本构建 Kernel：

```bash
# 构建全部 Kernel
bash scripts/build_kernel.sh

# 构建单个算子
bash scripts/build_kernel.sh --single_op=unique_v2

# Debug 模式构建
bash scripts/build_kernel.sh --build_type=Debug
```

### 3.5 Kernel 安装

将编译好的 Kernel 包安装到 CANN OPP 目录：

```bash
cd build_out
bash ../scripts/install_kernel.sh --install-path=/path/to/opp

# 或使用环境变量
export ASCEND_CUSTOM_OPP_PATH=/path/to/custom/opp
bash ../scripts/install_kernel.sh
```

安装内容包括：
- `op_proto/` — 算子原型库
- `op_impl/` — Tiling 库 + Kernel 实现 + 二进制 Kernel
- `op_api/` — ACLNN API 库和头文件
- `framework/` — 算子支持列表
- `version.info` — 版本信息

---

## 4. 运行时工作原理

### 4.1 Python 包初始化

当 `import rampage` 时，`__init__.py` 执行以下操作：

1. **检测芯片型号**：通过 `Dsmi_dc_Func` 类调用 `libdrvdsmi_host.so` 获取芯片名称
2. **选择算子包路径**：
   - 普通架构 → `packages/vendors/customize`
   - Arch35（ascend910_95） → `packages/vendors/customize_arch35`
3. **设置环境变量**：将算子包路径设置到 `ASCEND_CUSTOM_OPP_PATH`
4. **初始化 OpAPI**：调用 `rampage._C._init_op_api_so_path()` 加载自定义 `libcust_opapi.so`

### 4.2 算子调用链路

以 `npu_unique` 算子为例，完整调用链路如下：

```
Python 调用层                    C++ 调用层                     Kernel 层
─────────────                  ────────────                   ──────────
rampage.npu_unique(tensor)
    │
    ▼
rampage.ops.npu_unique.py      
  UniqueFunction.forward()
    │
    ▼
rampage._C.npu_unique()  ──→  Unique.cpp: npu_unique()
                                   │
                                   ▼
                              EXEC_NPU_CMD_SYNC(aclnnUniqueV2, ...)
                                   │
                                   ▼
                              libcust_opapi.so                 
                              → aclnnUniqueV2GetWorkspaceSize()
                              → aclnnUniqueV2()
                                   │
                                   ▼
                              libcust_opmaster_rt2.0.so
                              → UniqueV2TilingFunc() ──→  Kernel: unique_v2()
                                                          (AscendC AICore)
```

---

## 5. 算子结构详解（以 UniqueV2 为例）

### 5.1 总览

一个完整的算子由以下 **六个** 文件组成：

| 层级 | 文件 | 作用 |
|------|------|------|
| **Kernel Host** | `kernels/op_host/unique_v2_tiling.h` | Tiling 数据结构定义 |
| **Kernel Host** | `kernels/op_host/unique_v2.cpp` | 算子定义（OpDef）+ Tiling 函数 + 形状/类型推导 |
| **Kernel Device** | `kernels/op_kernel/unique_v2.h` | AscendC Kernel 类实现 |
| **Kernel Device** | `kernels/op_kernel/unique_v2.cpp` | Kernel 入口函数 |
| **C++ 封装** | `rampage/csrc/Unique.cpp` | torch Tensor 输入 → aclnn API 调用 |
| **Python 封装** | `rampage/ops/npu_unique.py` | Python 用户接口 |

### 5.2 Tiling 数据定义 (`kernels/op_host/unique_v2_tiling.h`)

Tiling 数据在 Host 侧计算，传递给 Device 侧 Kernel 使用：

```cpp
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UniqueV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);      // 数据总长度
    TILING_DATA_FIELD_DEF(uint32_t, shortBlockTileNum); // 短 block 的 tile 数
    TILING_DATA_FIELD_DEF(uint16_t, tileLength);        // 每个 tile 的长度
    TILING_DATA_FIELD_DEF(uint16_t, tailLength);        // 尾部长度
    TILING_DATA_FIELD_DEF(uint8_t, aivNum);             // AIV 核心数
    TILING_DATA_FIELD_DEF(uint8_t, blockNum);           // 使用的 block 数
    TILING_DATA_FIELD_DEF(uint8_t, shortBlockNum);      // 短 block 数量
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UniqueV2, UniqueV2TilingData)
}
```

**关键宏**：
- `BEGIN_TILING_DATA_DEF(name)` / `END_TILING_DATA_DEF` — 定义 Tiling 数据结构
- `TILING_DATA_FIELD_DEF(type, name)` — 定义字段
- `REGISTER_TILING_DATA_CLASS(OpName, TilingDataClass)` — 注册绑定

### 5.3 算子定义 (`kernels/op_host/unique_v2.cpp`)

此文件包含三部分：

#### (a) Tiling 函数

```cpp
namespace optiling {
static ge::graphStatus UniqueV2TilingFunc(gert::TilingContext* context) {
    UniqueV2TilingData tiling;
    // 1. 获取输入 shape
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    // 2. 计算 tiling 参数（分块策略）
    //    - totalLength, tileNum, blockNum 等
    // 3. 设置 tiling 数据
    tiling.set_totalLength(totalLength);
    // ...
    // 4. 设置 block 并行度
    context->SetBlockDim(blockNum);
    // 5. 保存 tiling 数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), ...);
    // 6. 设置 workspace 大小
    context->GetWorkspaceSizes(1)[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}
```

#### (b) 形状和类型推导

```cpp
namespace ge {
static ge::graphStatus UniqueV2InferShape(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static ge::graphStatus UniqueV2InferDtype(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}
```

#### (c) 算子定义（OpDef）

```cpp
namespace ops {
class UniqueV2 : public OpDef {
public:
    explicit UniqueV2(const char* name) : OpDef(name) {
        // 定义输入
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ...})
            .Format({ge::FORMAT_ND, ...});
        // 定义输出
        this->Output("output").ParamType(REQUIRED).DataType({...}).Format({...});
        this->Output("uniqueCnt").ParamType(REQUIRED).DataType({...}).Format({...});
        // 绑定推导函数
        this->SetInferShape(ge::UniqueV2InferShape);
        this->SetInferDataType(ge::UniqueV2InferDtype);
        // 绑定 Tiling 函数和支持的芯片
        this->AICore().SetTiling(optiling::UniqueV2TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(UniqueV2);  // 注册算子
}
```

### 5.4 Kernel 入口 (`kernels/op_kernel/unique_v2.cpp`)

```cpp
#include "kernel_operator.h"
#include "unique_v2.h"

extern "C" __global__ __aicore__ void unique_v2(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelUnique<DTYPE_INPUT> op(pipe);
    op.Init(input, output, uniqueCnt, workspace, ...);
    op.Process();
}
```

**关键宏和修饰符**：
- `extern "C"` — C 链接约定
- `__global__` — 全局 Kernel 入口
- `__aicore__` — 运行在 AICore 上
- `GM_ADDR` — Global Memory 地址类型
- `GET_TILING_DATA(var, ptr)` — 从 tiling buffer 中解析 tiling 数据
- `DTYPE_INPUT` — 由编译系统自动替换为实际数据类型

### 5.5 Kernel 实现 (`kernels/op_kernel/unique_v2.h`)

这是核心的 AscendC 并行计算逻辑，包含：
- `Init()` — 初始化 Global Memory Tensor、Workspace、Buffer
- `Process()` — 主处理流程：CopyIn → Sort → Unique → CopyOut
- 使用 `TPipe`、`TBuf`、`LocalTensor`、`GlobalTensor` 等 AscendC 编程原语
- 多核同步使用 `IBSet`/`IBWait`、`SyncAll` 等机制

### 5.6 C++ 封装 (`rampage/csrc/Unique.cpp`)

```cpp
#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor npu_unique(const at::Tensor& input) {
    TORCH_CHECK_NPU(input);
    if (input.numel() <= 1) {
        return at::Tensor(input).clone();
    }
    at::Tensor output = at::empty({input.numel()}, ...);
    at::Tensor uniqueCnt = at::empty({1}, ...);
    EXEC_NPU_CMD_SYNC(aclnnUniqueV2, input, output, uniqueCnt);
    int uniqueCount = uniqueCnt.item<int>();
    return output.narrow(0, 0, uniqueCount);
}
```

**关键宏**：
- `TORCH_CHECK_NPU(input)` — 检查输入 Tensor 是否在 NPU 上
- `EXEC_NPU_CMD_SYNC(aclnn_api, ...)` — 同步调用 ACLNN API

### 5.7 Python 封装 (`rampage/ops/npu_unique.py`)

```python
from torch.autograd import Function
import rampage._C

class UniqueFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        y = rampage._C.npu_unique(input_tensor)
        return y

npu_unique = UniqueFunction.apply
```

---

## 6. 如何添加一个新算子

以添加名为 `MyCustomOp` 的算子为例，该算子接受一个输入 Tensor，输出一个处理后的 Tensor。

### 6.1 步骤总览

```
Step 1: 编写 Kernel Host 侧代码（Tiling + OpDef）
Step 2: 编写 Kernel Device 侧代码（AscendC Kernel）
Step 3: 编写 C++ 封装（ACLNN 调用）
Step 4: 注册 PyBind11 绑定
Step 5: 编写 Python 封装
Step 6: 导出到包的 __init__.py
Step 7: 重新编译
```

### 6.2 Step 1：编写 Kernel Host 侧代码

#### 1a. 创建 Tiling 数据定义

**文件**: `kernels/op_host/my_custom_op_tiling.h`

```cpp
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MyCustomOpTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailLength);
    TILING_DATA_FIELD_DEF(uint8_t, blockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MyCustomOp, MyCustomOpTilingData)
}
```

#### 1b. 创建算子定义文件

**文件**: `kernels/op_host/my_custom_op.cpp`

```cpp
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "my_custom_op_tiling.h"

// =================== Tiling 函数 ===================
namespace optiling {
static ge::graphStatus MyCustomOpTilingFunc(gert::TilingContext* context)
{
    if (!context) return ge::GRAPH_FAILED;

    MyCustomOpTilingData tiling;

    // 获取输入 shape
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (!inputShape) return ge::GRAPH_FAILED;

    // 计算数据总长度
    uint32_t totalLength = 1;
    for (int i = 0; i < inputShape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= inputShape->GetStorageShape().GetDim(i);
    }

    // 获取平台信息
    const auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    // 计算分块策略
    constexpr uint32_t tileLength = 1024;
    uint32_t tileNum = (totalLength + tileLength - 1) / tileLength;
    uint8_t blockNum = tileNum >= coreNum ? coreNum : tileNum;
    uint32_t tailLength = totalLength % tileLength;

    // 设置 tiling 参数
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLength(tileLength);
    tiling.set_tailLength(tailLength);
    tiling.set_blockNum(blockNum);

    // 设置并行度
    context->SetBlockDim(blockNum);

    // 保存 tiling 数据
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

// =================== Shape/Dtype 推导 ===================
namespace ge {
static ge::graphStatus MyCustomOpInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (!inputShape || !outputShape) return GRAPH_FAILED;
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus MyCustomOpInferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

// =================== 算子注册 ===================
namespace ops {
class MyCustomOp : public OpDef {
public:
    explicit MyCustomOp(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::MyCustomOpInferShape);
        this->SetInferDataType(ge::MyCustomOpInferDtype);

        this->AICore().SetTiling(optiling::MyCustomOpTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MyCustomOp);
} // namespace ops
```

### 6.3 Step 2：编写 Kernel Device 侧代码

#### 2a. Kernel 实现

**文件**: `kernels/op_kernel/my_custom_op.h`

```cpp
#include "kernel_operator.h"
using namespace AscendC;

template<typename T>
class KernelMyCustomOp {
public:
    __aicore__ inline KernelMyCustomOp() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y,
        uint32_t totalLength, uint32_t tileNum,
        uint32_t tileLength, uint32_t tailLength,
        uint8_t blockNum)
    {
        // 计算当前 block 处理的数据范围
        uint32_t tilesPerBlock = tileNum / blockNum;
        uint32_t extraTiles = tileNum % blockNum;
        uint32_t startTile, endTile;
        if (GetBlockIdx() < extraTiles) {
            startTile = GetBlockIdx() * (tilesPerBlock + 1);
            endTile = startTile + tilesPerBlock + 1;
        } else {
            startTile = extraTiles * (tilesPerBlock + 1) +
                        (GetBlockIdx() - extraTiles) * tilesPerBlock;
            endTile = startTile + tilesPerBlock;
        }

        this->startOffset = startTile * tileLength;
        this->processLength = (endTile - startTile) * tileLength;
        if (GetBlockIdx() == blockNum - 1 && tailLength > 0) {
            this->processLength = this->processLength - tileLength + tailLength;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + this->startOffset, this->processLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + this->startOffset, this->processLength);

        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->processLength / TILE_LENGTH;
        uint32_t tailLen = this->processLength % TILE_LENGTH;

        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i, TILE_LENGTH);
            Compute(TILE_LENGTH);
            CopyOut(i, TILE_LENGTH);
        }
        if (tailLen > 0) {
            CopyIn(loopCount, tailLen);
            Compute(tailLen);
            CopyOut(loopCount, tailLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // === 在此处实现你的计算逻辑 ===
        // 示例：y = x * 2
        Muls(yLocal, xLocal, (T)2, len);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[progress * TILE_LENGTH], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    static constexpr uint32_t TILE_LENGTH = 1024;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<T> xGm, yGm;

    uint32_t startOffset;
    uint32_t processLength;
};
```

#### 2b. Kernel 入口

**文件**: `kernels/op_kernel/my_custom_op.cpp`

```cpp
#include "kernel_operator.h"
#include "my_custom_op.h"

extern "C" __global__ __aicore__ void my_custom_op(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMyCustomOp<DTYPE_X> op;
    op.Init(x, y,
            tiling_data.totalLength,
            tiling_data.tileNum,
            tiling_data.tileLength,
            tiling_data.tailLength,
            tiling_data.blockNum);
    op.Process();
}
```

> **注意**：Kernel 入口函数名（`my_custom_op`）必须是算子类名 `MyCustomOp` 的下划线命名形式。`DTYPE_X` 对应 OpDef 中 Input("x") 的数据类型占位符，编译系统会自动替换。

### 6.4 Step 3：编写 C++ 封装

**文件**: `rampage/csrc/MyCustomOp.cpp`

```cpp
#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor npu_my_custom_op(const at::Tensor& input)
{
    TORCH_CHECK_NPU(input);

    // 创建输出 Tensor（与输入同 shape、同 dtype、同 device）
    at::Tensor output = at::empty_like(input);

    // 调用 ACLNN API（名称格式：aclnn + 算子类名）
    EXEC_NPU_CMD(aclnnMyCustomOp, input, output);

    return output;
}
```

### 6.5 Step 4：注册 PyBind11 绑定

**修改文件**: `include/csrc/functions.h`

```cpp
// 在文件中添加函数声明
at::Tensor npu_my_custom_op(const at::Tensor& input);
```

**修改文件**: `rampage/csrc/pybind.cpp`

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_init_op_api_so_path", &init_op_api_so_path);
    m.def("npu_unique", &npu_unique);
    m.def("npu_my_custom_op", &npu_my_custom_op);  // 新增
}
```

### 6.6 Step 5：编写 Python 封装

**文件**: `rampage/ops/npu_my_custom_op.py`

```python
from torch.autograd import Function
import rampage._C


class MyCustomOpFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        output = rampage._C.npu_my_custom_op(input_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 如需支持反向传播，在此实现
        # 否则可省略 backward 方法
        return grad_output * 2  # 对应 forward 中的 x * 2


npu_my_custom_op = MyCustomOpFunction.apply
```

### 6.7 Step 6：导出到 `__init__.py`

**修改文件**: `rampage/__init__.py`

```python
__all__ = [
    "npu_unique",
    "npu_my_custom_op",  # 新增
]

# ... 已有代码 ...

from .ops.npu_unique import npu_unique
from .ops.npu_my_custom_op import npu_my_custom_op  # 新增
```

### 6.8 Step 7：重新编译

```bash
# 开发模式 - 全量编译
pip install -e .

# 或仅编译新增算子的 Kernel
pip install -e . --install-option="--kernel-name=my_custom_op"
```

### 6.9 新增算子文件清单

| # | 文件路径 | 操作 |
|---|---------|------|
| 1 | `kernels/op_host/my_custom_op_tiling.h` | **新建** |
| 2 | `kernels/op_host/my_custom_op.cpp` | **新建** |
| 3 | `kernels/op_kernel/my_custom_op.h` | **新建** |
| 4 | `kernels/op_kernel/my_custom_op.cpp` | **新建** |
| 5 | `rampage/csrc/MyCustomOp.cpp` | **新建** |
| 6 | `include/csrc/functions.h` | **修改** — 添加函数声明 |
| 7 | `rampage/csrc/pybind.cpp` | **修改** — 添加 PyBind 注册 |
| 8 | `rampage/ops/npu_my_custom_op.py` | **新建** |
| 9 | `rampage/__init__.py` | **修改** — 添加导出 |

> **注意**：`kernels/op_host/` 和 `kernels/op_kernel/` 下的文件会被 CMake 通过 `file(GLOB ...)` 自动扫描，**无需修改 CMakeLists.txt**。`rampage/csrc/` 下的文件同理。

---

## 7. 常见问题

### Q1: 如何查看构建产物？

编译完成后，算子包位于 `rampage/packages/vendors/customize/` 目录下：
```
packages/vendors/customize/
├── op_proto/           # 算子原型库
├── op_impl/            # Tiling 库 + Kernel 实现 + 二进制
├── op_api/             # ACLNN API 库和头文件
├── framework/          # npu_supported_ops.json
└── version.info
```

### Q2: 如何支持新的芯片型号？

在 `kernels/op_host/xxx.cpp` 的 OpDef 中添加：
```cpp
this->AICore().AddConfig("ascend_new_chip");
```
同时在 `CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT` 列表中添加新芯片标识。

### Q3: 如何为 Arch35 架构添加算子？

1. 在 `kernels_arch35/op_host/` 和 `kernels_arch35/op_kernel/` 下添加算子文件
2. 需要在 `kernels_arch35/CMakeLists.txt` 中添加对应的构建逻辑（参考 `kernels/CMakeLists.txt`）
3. 构建时设置 `USE_ARCH35=true`

### Q4: Debug 模式如何编译？

```bash
# 通过 setup.py
pip install -e . --install-option="--release"  # 不加此选项则默认 Debug

# 通过 build_kernel.sh
bash scripts/build_kernel.sh --build_type=Debug
```

### Q5: CANN 路径如何配置？

优先级从高到低：
1. 环境变量 `ASCEND_AICPU_PATH`
2. `CMakePresets.json` 中的 `ASCEND_CANN_PACKAGE_PATH`
3. 默认路径 `/usr/local/Ascend/latest`

---

## 8. 附录：核心依赖

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| CMake | ≥ 3.19.0 | 构建系统 |
| Python | 3.x | 包管理和辅助脚本 |
| PyTorch | ≥ 2.0 | 深度学习框架 |
| torch_npu | 匹配 PyTorch 版本 | 昇腾 NPU 后端 |
| CANN Toolkit | 对应芯片版本 | 昇腾计算框架 |
| GCC/G++ | ≥ 4.8.5 | C++ 编译器 |
