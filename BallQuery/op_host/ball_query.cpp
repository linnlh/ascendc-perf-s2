
#include "ball_query_tiling.h"
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"


namespace optiling {

static int32_t getAlignPadding(int32_t length, int32_t align) {
    int32_t padding = 0;
    while ((length + padding) % align)
        padding += 1;
    return padding;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto maxBlockNum = ascendcPlatform.GetCoreNumAiv();
  int32_t blockNum = 1;

  // Set tiling.
  BallQueryTilingData tiling;
  auto xyzShape = context->GetInputShape(0);
  auto centerXyzShape = context->GetInputShape(1);
  auto xyzBatchCntShape = context->GetInputShape(2);
  auto centerXyzBatchCntShape = context->GetInputShape(3);
  auto attrs = context->GetAttrs();
    
  int64_t totalBatchSize = 0;
  int64_t batchSize = 0;
  int64_t xyzNum = 0;
  int64_t centerNum = 0;
  if (xyzBatchCntShape == nullptr) {
    xyzNum = xyzShape->GetStorageShape().GetDim(1);
    centerNum = centerXyzShape->GetStorageShape().GetDim(1);
    totalBatchSize = xyzShape->GetStorageShape().GetDim(0);
    blockNum = maxBlockNum;
    while(totalBatchSize % blockNum) {
        blockNum -= 1;
    }
    batchSize = totalBatchSize / blockNum;
    context->SetBlockDim(blockNum);
    tiling.set_isStack(false);
    if (context->GetInputTensor(0)->GetDataType() == ge::DT_FLOAT) {
        batchSize = totalBatchSize;
        blockNum = maxBlockNum;
        while (centerNum % blockNum) {
            blockNum -= 1;
        }
        context->SetBlockDim(blockNum);
        tiling.set_isStack(false);
    }
  }
  else {
    totalBatchSize = xyzBatchCntShape->GetStorageShape().GetDim(0);
    batchSize = xyzBatchCntShape->GetStorageShape().GetDim(0);
    int64_t totalXyzNum = xyzShape->GetStorageShape().GetDim(0);
    int64_t totalCenterXyzNum = centerXyzShape->GetStorageShape().GetDim(0);
    xyzNum = 16;
    centerNum = 32;
    context->SetBlockDim(1);
    tiling.set_isStack(true);
    tiling.set_totalXyzNum(totalXyzNum);
    tiling.set_totalCenterXyzNum(totalCenterXyzNum);
  }

  const float* minRadius = attrs->GetAttrPointer<float>(0);
  const float* maxRadius = attrs->GetAttrPointer<float>(1);
  const int* numSample = attrs->GetAttrPointer<int>(2);
  int32_t indicePadding = getAlignPadding(*numSample, 8);

  // Set tiling.
  tiling.set_batchSize(batchSize);
  tiling.set_xyzNum(xyzNum);
  tiling.set_centerNum(centerNum);
  tiling.set_numSample(*numSample);
  tiling.set_minRadius2((*minRadius) * (*minRadius));
  tiling.set_maxRadius2((*maxRadius) * (*maxRadius));
  tiling.set_indicePadding(indicePadding);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class BallQuery : public OpDef {
public:
    explicit BallQuery(const char* name) : OpDef(name)
    {
        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("xyz_batch_cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz_batch_cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min_radius").Float();
        this->Attr("max_radius").Float();
        this->Attr("sample_num").Int();

        // this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(BallQuery);
}
