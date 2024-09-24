#include <unordered_map>

#include "depth_to_space_tiling.h"
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"


namespace optiling {

static int GetAlignPadding(int32_t val, int32_t align) {
    int padding = 0;
    while ((val + padding) % align) {
        padding += 1;
    }
    return padding;
}

static int GetMode(const char* dataFormat, const char* mode) {
    if ((strcmp(dataFormat, "NHWC")) == 0) {
        if ((strcmp(mode, "DCR")) == 0) {
            return 0;
        }
        else {
            return 1;
        }
    }
    else {
        if ((strcmp(mode, "DCR")) == 0) {
            return 2;
        }
        else {
            return 3;
        }
    }
}


const int32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();

    auto shape = context->GetInputShape(0)->GetStorageShape();
    int32_t N = shape.GetDim(0);
    int32_t H = shape.GetDim(1);
    int32_t W = shape.GetDim(2);
    int32_t C = shape.GetDim(3); 


    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxBlockNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    // Set tiling data
    DepthToSpaceTilingData tiling;
    auto shape = context->GetInputShape(0)->GetStorageShape();
    auto attrs = context->GetAttrs();

    const int64_t* blockSize = attrs->GetInt(0);
    const char* transMode = attrs->GetStr(1);
    const char* dataFormat = attrs->GetStr(2);

    int32_t blockDim = shape.GetDim(0);
    int32_t N, H, W, C;
    int32_t batchSize = 1;
    int32_t strideSize, padedStrideSize;
    if (strcmp(dataFormat, "NHWC") == 0) {
        N = 1;
        H = shape.GetDim(1);
        W = shape.GetDim(2);
        C = shape.GetDim(3);
        batchSize = shape.GetDim(0) * H;
        if (strcmp(transMode, "DCR") == 0) {
            blockDim = maxBlockNum;
            batchSize = batchSize / blockDim;
            remainBatch = batchSize % blockDim;
            strideSize = C / (*blockSize);
            padedStrideSize = strideSize + GetAlignPadding(strideSize, 8);
            int32_t tileLength = std::max(64, W);
            int32_t lastTile = W % tileLength;
            tiling.set_tileLength(tileLength);
            tiling.set_lastTile(lastTile);
        }
    }
    else {
        N = 1;
        // N = shape.GetDim(0);
        H = shape.GetDim(2);
        W = shape.GetDim(3);
        C = shape.GetDim(1);
    }

    int32_t mode = GetMode(dataFormat, transMode);
    tiling.set_N(N);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_C(C);
    tiling.set_blockSize((int32_t)(*blockSize));
    tiling.set_mode(mode);
    tiling.set_batchSize(batchSize);
    tiling.set_strideSize(strideSize);
    tiling.set_padedStrideSize(padedStrideSize);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );

    // Set context
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    context->SetBlockDim(blockDim);
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
class DepthToSpace : public OpDef {
public:
    explicit DepthToSpace(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("block_size").Int();
        this->Attr("mode").AttrType(OPTIONAL).String("DCR");
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(DepthToSpace);
}
