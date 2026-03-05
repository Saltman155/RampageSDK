# RampageSDK 算子库技术文档

## 1. 项目概述

RampageSDK 是一个面向华为昇腾（Ascend）NPU 的自定义算子库，旨在为自动驾驶系统提供加速支持。它基于 CANN（Compute Architecture for Neural Networks）工具链，通过 AscendC 编程模型编写高性能自定义算子，并通过 PyTorch + torch_npu 接口将算子暴露给 Python 用户使用。

### 1.1 核心特性

- 支持多种昇腾芯片架构：ascend910b、ascend910、ascend310p、ascend910_93、ascend910_95（A5）
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
├── ci/                         # ci构建脚本
├── kernels/                    # 标准架构算子 Kernel 源码
│   ├── CMakeLists.txt          # Kernel 构建逻辑（Stage 0 + Stage 1）
│   ├── op_host/                # Host 侧代码（算子定义、Tiling 函数、形状推导）
│   └── op_kernel/              # Device 侧代码（AscendC Kernel 实现）
├── kernels_arch35/             # Arch35 架构（ascend910_95）专用 Kernel（结构同 kernels/）
│   └── CMakeLists.txt          # Arch35 专属算子构建逻辑
├── rampage/                    # Python 包源码
│   ├── __init__.py             # 包初始化、环境配置、算子导出
│   ├── get_chip_info.py        # 芯片型号检测（区分 arch35）
│   ├── csrc/                   # C++ 扩展源码
│   │   ├── CMakeLists.txt      # C++ 扩展构建（Stage 2，生成 _C.so）
│   │   ├── pybind.cpp          # PyBind11 绑定入口
│   │   └── OpApiCommon.cpp     # OpAPI 通用工具（参数序列化、哈希计算）
│   └── ops/                    # Python 算子接口
├── include/                    # 公共头文件
│   ├── csrc/                   # C++ 扩展相关头文件
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

RampageSDK 通过标准的 Python `setuptools` 进行构建和安装。执行 `python setup.py develop`（开发模式）或 `python setup.py bdist_wheel`（正式安装）即可触发完整构建。

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
# 开发模式构建（推荐开发时使用）
python setup.py develop

# 正式构建
python setup.py bdist_wheel

# 只构建单个算子 Kernel（开发模式）
python setup.py develop --kernel-name="DeformableConv2d;MultiScaleDeformableAttn"

# 使用 arch35 架构（A5算子）
USE_ARCH35=true python setup.py bdist_wheel

```

## 4. 框架编译构建工作原理

请参考 [框架详解](docs/FRAME.md)。



## 5. 算子结构详解

请参考 [算子结构详解（以 UniqueV2 为例）](docs/OP_SAMPLE.md)。


## 6. 如何添加一个新算子

请参考 [算子结构详解（以 MyCustomOp 为例）](docs/OP_ADD.md)。



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
