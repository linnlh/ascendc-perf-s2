
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormV2TilingData)
  TILING_DATA_FIELD_DEF(int32_t, batchSize);
  TILING_DATA_FIELD_DEF(int32_t, groupSize);
  TILING_DATA_FIELD_DEF(int32_t, chunkSize);
  TILING_DATA_FIELD_DEF(int32_t, lastChunkSize);
  TILING_DATA_FIELD_DEF(int32_t, chunkNum);
  TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormV2, GroupNormV2TilingData)
}
