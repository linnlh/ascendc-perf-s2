
#include "group_norm_v2_tiling.h"
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static constexpr int32_t MAX_TILE_SIZE = 4096;
static constexpr int32_t CHUNK_NUM = 16;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxBlockNum = ascendcPlatform.GetCoreNumAiv();
    int32_t blockNum = maxBlockNum;
    if (context->GetInputTensor(0)->GetDataType() == ge::DT_FLOAT16)
        blockNum = 1;
    
    GroupNormV2TilingData tiling;
    auto shape = context->GetInputShape(0)->GetStorageShape();
    auto attrs = context->GetAttrs();
    const int32_t* numGroups = attrs->GetAttrPointer<int>(0);
    const float* eps = attrs->GetAttrPointer<float>(2);

    int32_t groupSize = shape.GetDim(1) / (*numGroups) * shape.GetDim(2);
    if (shape.GetDimNum() == 4)
        groupSize *= shape.GetDim(3);
    if (context->GetInputTensor(0)->GetDataType() == ge::DT_FLOAT &&
        groupSize > 1024) {
        int32_t batchSize = shape.GetDim(0) * (*numGroups);
        while(groupSize % blockNum) {
            blockNum -= 1;
        }
        int32_t blockSize = groupSize / blockNum;
        int32_t chunkNum = CHUNK_NUM;
        int32_t chunkSize = blockSize / chunkNum;
        int32_t lastChunkSize = chunkSize;
        if (blockSize % chunkNum) {
            chunkSize = blockSize / (chunkNum - 1);
            lastChunkSize = blockSize % (chunkNum - 1);
        }
        tiling.set_batchSize(batchSize);
        tiling.set_groupSize(groupSize);
        tiling.set_chunkSize(chunkSize);
        tiling.set_chunkNum(chunkNum);
        tiling.set_lastChunkSize(lastChunkSize);
        tiling.set_eps(*eps);
        // context->SetTilingKey(1);
    }
    else {
        int32_t totalBatchSize = shape.GetDim(0) * (*numGroups);
        while (totalBatchSize % blockNum) {
            blockNum -= 1;
        }
        int32_t batchSize = totalBatchSize / blockNum;
        int32_t chunkSize = groupSize;
        int32_t chunkNum = 1;
        int32_t lastChunkSize = groupSize;
        if (groupSize > MAX_TILE_SIZE) {
            chunkSize = MAX_TILE_SIZE;
            chunkNum = groupSize / chunkSize;
            lastChunkSize = MAX_TILE_SIZE;
            if (groupSize % chunkSize != 0) {
                chunkNum += 1;
                lastChunkSize = groupSize % chunkSize;
            }
        }

        tiling.set_batchSize(batchSize);
        tiling.set_groupSize(groupSize);
        tiling.set_chunkSize(chunkSize);
        tiling.set_chunkNum(chunkNum);
        tiling.set_lastChunkSize(lastChunkSize);
        tiling.set_eps(*eps);
    }

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrSize = 4096;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    
    context->SetBlockDim(blockNum);
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
class GroupNormV2 : public OpDef {
public:
    explicit GroupNormV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_groups").Int();
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        this->Attr("eps").AttrType(OPTIONAL).Float(0.0001);
        this->Attr("is_training").AttrType(OPTIONAL).Bool(true);

        // this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(GroupNormV2);
}
