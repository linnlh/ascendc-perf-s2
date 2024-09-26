
#include "register/tilingdata_base.h"

#include <string>

namespace optiling {
BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
  TILING_DATA_FIELD_DEF(int32_t, N);
  TILING_DATA_FIELD_DEF(int32_t, H);
  TILING_DATA_FIELD_DEF(int32_t, W);
  TILING_DATA_FIELD_DEF(int32_t, C);
  TILING_DATA_FIELD_DEF(int32_t, blockSize);
  TILING_DATA_FIELD_DEF(int32_t, mode);
  // NHWC, DCR
  TILING_DATA_FIELD_DEF(int32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData)
}
