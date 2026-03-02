
### 编译框架详解




┌─────────────────────────────────────────────────────────────────┐ </br>
│                    完整构建流程                                  │ </br>
├─────────┬───────────────────────────────────────────────────────┤ </br>
│ Stage 0 │ 编译 op_host → 运行 opbuild → 生成自动代码              │ </br>
├─────────┼───────────────────────────────────────────────────────┤ </br>
│ Stage 1 │ 编译 proto/tiling/opapi 库 → 生成 ops info             │ </br>
│         │ → 生成 impl Python → 二进制编译                        │ </br>
├─────────┼───────────────────────────────────────────────────────┤ </br>
│ Stage 2 │ 编译 PyTorch C++ 扩展 _C.so                            │ </br>
└─────────┴───────────────────────────────────────────────────────┘ </br>
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

### CMake 配置体系

#### `CMakePresets.json` — 预设配置

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

#### `cmake/config.cmake` — 路径和环境检测

- 从环境变量 `ASCEND_AICPU_PATH` 或默认路径 `/usr/local/Ascend/latest` 检测 CANN 安装路径
- 通过 `ASCEND_CANN_PACKAGE_PATH` 的软链接定位编译器路径
- 检测系统架构（aarch64/x86_64）
- 设置自动代码生成目录 `ASCEND_AUTOGEN_PATH`

#### `cmake/intf.cmake` — 编译接口库

定义了一个 INTERFACE 库 `intf_pub`，统一管理编译选项：

- **编译选项**: `-fPIC`、`-fvisibility=hidden`、C++11 标准
- **安全选项**: `-fstack-protector-strong`、`-Wl,-z,relro`、`-Wl,-z,now`
- **Release**: `-O2` + `_FORTIFY_SOURCE=2`
- **Debug**: `-O0 -g -ftrapv -fstack-check`
- **ABI**: `_GLIBCXX_USE_CXX11_ABI=0`

#### `cmake/func.cmake` — 构建函数库

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

### 独立 Kernel 构建

除了通过 `setup.py` 构建外，也可以使用独立脚本构建 Kernel：

```bash
# 构建全部 Kernel
bash scripts/build_kernel.sh

# 构建单个算子
bash scripts/build_kernel.sh --single_op=unique_v2

# Debug 模式构建
bash scripts/build_kernel.sh --build_type=Debug
```

### Kernel 安装

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


当 `import rampage` 时，`__init__.py` 执行以下操作：

1. **检测芯片型号**：通过 `Dsmi_dc_Func` 类调用 `libdrvdsmi_host.so` 获取芯片名称
2. **选择算子包路径**：
   - 普通架构 → `packages/vendors/customize`
   - Arch35（ascend910_95） → `packages/vendors/customize_arch35`
3. **设置环境变量**：将算子包路径设置到 `ASCEND_CUSTOM_OPP_PATH`
4. **初始化 OpAPI**：调用 `rampage._C._init_op_api_so_path()` 加载自定义 `libcust_opapi.so`

### 算子调用链路

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