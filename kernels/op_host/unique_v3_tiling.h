#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UniqueV3TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, shortBlockTileNum);
    TILING_DATA_FIELD_DEF(uint16_t, tileLength);
    TILING_DATA_FIELD_DEF(uint16_t, tailLength);
    TILING_DATA_FIELD_DEF(uint8_t, aivNum);
    TILING_DATA_FIELD_DEF(uint8_t, blockNum);
    TILING_DATA_FIELD_DEF(uint8_t, shortBlockNum);
    TILING_DATA_FIELD_DEF(bool, flagInverse);
    TILING_DATA_FIELD_DEF(bool, flagCounts);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UniqueV3, UniqueV3TilingData)
}
