

### 算子开发详解（以 UniqueV2 为例）

一个完整的算子由以下 **六个** 文件组成：

| 层级 | 文件 | 作用 |
|------|------|------|
| **Kernel Host** | `kernels/op_host/unique_v2_tiling.h` | Tiling 数据结构定义 |
| **Kernel Host** | `kernels/op_host/unique_v2.cpp` | 算子定义（OpDef）+ Tiling 函数 + 形状/类型推导 |
| **Kernel Device** | `kernels/op_kernel/unique_v2.h` | AscendC Kernel 类实现 |
| **Kernel Device** | `kernels/op_kernel/unique_v2.cpp` | Kernel 入口函数 |
| **C++ 封装** | `rampage/csrc/Unique.cpp` | torch Tensor 输入 → aclnn API 调用 |
| **Python 封装** | `rampage/ops/npu_unique.py` | Python 用户接口 |

### Tiling 数据定义 (`kernels/op_host/unique_v2_tiling.h`)

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

### 算子定义 (`kernels/op_host/unique_v2.cpp`)

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

### Kernel 入口 (`kernels/op_kernel/unique_v2.cpp`)

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

### Kernel 实现 (`kernels/op_kernel/unique_v2.h`)

这是核心的 AscendC 并行计算逻辑，包含：
- `Init()` — 初始化 Global Memory Tensor、Workspace、Buffer
- `Process()` — 主处理流程：CopyIn → Sort → Unique → CopyOut
- 使用 `TPipe`、`TBuf`、`LocalTensor`、`GlobalTensor` 等 AscendC 编程原语
- 多核同步使用 `IBSet`/`IBWait`、`SyncAll` 等机制

### C++ 封装 (`rampage/csrc/Unique.cpp`)

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

### Python 封装 (`rampage/ops/npu_unique.py`)

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