/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "bev_pool_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
using namespace AscendC;


namespace optiling {

static ge::graphStatus TilingForBEVPoolV1(gert::TilingContext* context)
{
    // 这里可以根据输入输出的shape和属性来计算出tiling参数，填充到tiling_data中
    BevPoolV1TilingData tiling;
    tiling.set_totalLength(0);
    tiling.set_N(0);
    tiling.set_B(0);
    tiling.set_D(0);
    tiling.set_H(0);
    tiling.set_W(0);
    tiling.set_C(0);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {

static graphStatus InferShapeForBEVPoolV1(gert::InferShapeContext* context)
{
    // 这里可以根据输入的shape和属性来推断输出的shape，填充到output_shapes中
    return GRAPH_SUCCESS;
}

} // namespace ge



namespace ops {
class BEVPoolV1 : public OpDef {
public:
    explicit BEVPoolV1(const char* name) : OpDef(name)
    {
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("geom_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("b").AttrType(REQUIRED).Int();
        this->Attr("d").AttrType(REQUIRED).Int();
        this->Attr("h").AttrType(REQUIRED).Int();
        this->Attr("w").AttrType(REQUIRED).Int();
        this->Attr("c").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForBEVPoolV1);

        this->AICore().SetTiling(optiling::TilingForBEVPoolV1);
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(BEVPoolV1);

} // namespace ops
