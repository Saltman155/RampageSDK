## 新增算子教程（以添加名为 `MyCustomOp` 的算子为例）

以添加名为 `MyCustomOp` 的算子为例，该算子接受一个输入 Tensor，输出一个处理后的 Tensor。

### 步骤总览

```
Step 1: 编写 Kernel Host 侧代码（Tiling + OpDef）
Step 2: 编写 Kernel Device 侧代码（AscendC Kernel）
Step 3: 编写 C++ 封装（ACLNN 调用）
Step 4: 注册 PyBind11 绑定
Step 5: 编写 Python 封装
Step 6: 导出到包的 __init__.py
Step 7: 重新编译
```

### Step 1：编写 Kernel Host 侧代码

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

### Step 2：编写 Kernel Device 侧代码

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

### Step 3：编写 C++ 封装

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

### Step 4：注册 PyBind11 绑定

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

### Step 5：编写 Python 封装

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

### Step 6：导出到 `__init__.py`

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

### Step 7：重新编译

```bash
# 开发模式 - 全量编译
pip install -e .

# 或仅编译新增算子的 Kernel
pip install -e . --install-option="--kernel-name=my_custom_op"
```

### 新增算子文件清单

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