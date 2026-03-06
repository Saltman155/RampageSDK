#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "unique_v3_tiling.h"

constexpr size_t SYS_RSVD_WS_SIZE = 16 * 1024 * 1024;
constexpr size_t BYTE_PER_BLK = 32;
constexpr size_t EVENTID_MAX = 8;

namespace optiling {
static ge::graphStatus UniqueV3TilingFunc(gert::TilingContext* context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }
    UniqueV3TilingData tiling;

    constexpr uint16_t tileLength = 8192;
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (!inputShape) {
        return ge::GRAPH_FAILED;
    }
    const uint8_t dimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t totalLength = 1;
    for (int i = 0; i < dimNum; i++) {
        totalLength *= inputShape->GetStorageShape().GetDim(i);
    }
    const uint32_t tileNum = (totalLength + tileLength - 1) / tileLength;
    const uint16_t tailLength = totalLength % tileLength;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint8_t blockNum = tileNum >= aivNum ? aivNum : tileNum;
    const uint32_t shortBlockTileNum = tileNum / blockNum;
    const uint8_t longBlockNum = tileNum % blockNum;
    const uint8_t shortBlockNum = blockNum - longBlockNum;

    const bool *flagInversePtr = context->GetAttrs()->GetBool(0);
    const bool *flagCountsPtr = context->GetAttrs()->GetBool(1);
    const bool flagInverse = flagInversePtr ? *flagInversePtr : false;
    const bool flagCounts = flagCountsPtr ? *flagCountsPtr : false;

    tiling.set_totalLength(totalLength);
    tiling.set_shortBlockTileNum(shortBlockTileNum);
    tiling.set_tileLength(tileLength);
    tiling.set_tailLength(tailLength);
    tiling.set_aivNum(aivNum);
    tiling.set_blockNum(blockNum);
    tiling.set_shortBlockNum(shortBlockNum);
    tiling.set_flagInverse(flagInverse);
    tiling.set_flagCounts(flagCounts);

    context->SetBlockDim(blockNum);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // Workspace for IBSet/IBWait up to 8 times, and 2 times full data.
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto&& currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t usrSize = (blockNum * BYTE_PER_BLK * EVENTID_MAX + aivNum * BYTE_PER_BLK + BYTE_PER_BLK) +
                     (blockNum + BYTE_PER_BLK - 1) / BYTE_PER_BLK * BYTE_PER_BLK +
                     (tileNum * tileLength) * 2 * sizeof(float) * 2;

    // counts 统计count的临时空间 + 每个block头尾的值 + 每个block的unique个数
    // 这里blockNum也要对齐到32字节，因为后面要拷贝到UE上操作
    size_t countSize = (tileNum * tileLength) * sizeof(int32_t) + ((blockNum + BYTE_PER_BLK - 1) / BYTE_PER_BLK * BYTE_PER_BLK) * sizeof(uint32_t) * 3;
    // inverse 反向索引的临时空间 = 元素数量 x 2 （idx和value） x sizeof(int32_t) x 2 （两个buffer交替使用） + 每个block头尾的值 + 每个block的unique个数
    size_t inverseSize = (tileNum * tileLength) *sizeof(int32_t) * 2 * 2 + ((blockNum + BYTE_PER_BLK - 1) / BYTE_PER_BLK * BYTE_PER_BLK) * sizeof(uint32_t) * 3;

    currentWorkspace[0] = usrSize + countSize + inverseSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus UniqueV3InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (!x1_shape || !y_shape) {
        return GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus UniqueV3InferDtype(gert::InferDataTypeContext* context)
{
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class UniqueV3 : public OpDef {
public:
    explicit UniqueV3(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("uniqueCnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("inverse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("counts")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("flag_inverse").AttrType(OPTIONAL).Bool(false);
        this->Attr("flag_counts").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::UniqueV3InferShape);
        this->SetInferDataType(ge::UniqueV3InferDtype);

        this->AICore().SetTiling(optiling::UniqueV3TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(UniqueV3);
} // namespace ops
