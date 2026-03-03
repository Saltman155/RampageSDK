/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef BEV_POOL_V1_TILING_H
#define BEV_POOL_V1_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BevPoolV1TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, D);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, C);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BEVPoolV1, BevPoolV1TilingData)
}

#endif  // BEV_POOL_V1_TILING_H
