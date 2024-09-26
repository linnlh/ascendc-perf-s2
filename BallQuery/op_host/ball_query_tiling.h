
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BallQueryTilingData)
  TILING_DATA_FIELD_DEF(int64_t, batchSize);
  TILING_DATA_FIELD_DEF(int64_t, xyzNum);
  TILING_DATA_FIELD_DEF(int64_t, centerNum);
  TILING_DATA_FIELD_DEF(bool, isStack);
  TILING_DATA_FIELD_DEF(int64_t, numSample);
  TILING_DATA_FIELD_DEF(float, minRadius2);
  TILING_DATA_FIELD_DEF(float, maxRadius2);
  TILING_DATA_FIELD_DEF(int64_t, totalXyzNum);
  TILING_DATA_FIELD_DEF(int64_t, totalCenterXyzNum);
  TILING_DATA_FIELD_DEF(int32_t, indicePadding);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BallQuery, BallQueryTilingData)
}
