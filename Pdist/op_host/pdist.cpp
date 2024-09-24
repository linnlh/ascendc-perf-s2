#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
using namespace platform_ascendc;
const int BLOCK_SIZE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
	auto ascendcPlatform = PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();

    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t N = shape.GetDim(0);
    uint32_t M = shape.GetDim(1);
    auto attrs = context->GetAttrs();
    const float* p = attrs->GetAttrPointer<float>(0);


    uint32_t typeLength;
    auto dt = context->GetInputDesc(0)->GetDataType();
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        typeLength = 2;
    }
    else {
        typeLength = 4;
    }

    uint32_t totalBatch = N * (N - 1) / 2;
    uint32_t totalBatchByte = totalBatch * typeLength;
    uint32_t totalBatchByteAlign32 = (totalBatchByte + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    coreNum = (coreNum <  totalBatchByteAlign32 / BLOCK_SIZE) ? coreNum : totalBatchByteAlign32 / BLOCK_SIZE;
    coreNum = (coreNum >= 1) ? coreNum : 1;

    uint32_t everyCoreBlockNum = totalBatchByteAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (totalBatchByteAlign32 / BLOCK_SIZE) % coreNum;

    uint32_t coreBatch = everyCoreBlockNum * BLOCK_SIZE / typeLength;
    uint32_t tailBatch = tailBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigCoreNum = tailBlockNum == 0 ? coreNum : tailBlockNum;
    

    PdistTilingData tiling;
    tiling.set_N(N);
    tiling.set_M(M);
    tiling.set_p(*p);
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_smallCoreBatch(coreBatch);
    tiling.set_bigCoreBatch(coreBatch + BLOCK_SIZE / typeLength);

    context->SetBlockDim(coreNum);
    // 尚未实现 fp16 的多核逻辑，退化为单核的情况
    if (typeLength == 2)
        context->SetBlockDim(1);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    // currentWorkspace[0] = 0;
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
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("p").AttrType(OPTIONAL).Float(2.0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Pdist);
}
