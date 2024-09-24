#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template<typename T>
class KernelGroupNormV2 {
public:
    __aicore__ inline KernelGroupNormV2() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR gamma,
        GM_ADDR beta,
        GM_ADDR y,
        GM_ADDR mean,
        GM_ADDR rstd,
        GroupNormV2TilingData& tiling
    ) {
        // Init global memory.
        InitTiling(tiling);
        int32_t blockSize = batchSize * groupSize;
        Gm_x.SetGlobalBuffer((__gm__ T *)x + GetBlockIdx() * blockSize, blockSize);
        Gm_y.SetGlobalBuffer((__gm__ T *)y + GetBlockIdx() * blockSize, blockSize);

        // Init buffer.
        pipe.InitBuffer(Q_x, BUFFER_NUM, chunkSize * sizeof(T));
        pipe.InitBuffer(Q_y, BUFFER_NUM, chunkSize * sizeof(T));
        pipe.InitBuffer(B_work, chunkSize);
        // PrintParam();
    }
    __aicore__ inline void Process() {
        LocalTensor<T> workspace = B_work.Get<T>();
        for (int32_t i = 0; i < batchSize; i++) {
            int32_t offset = i * groupSize;

            LocalTensor<T> xChunk;
            float meanVal = 0.0f;
            for (int32_t j = 0; j < chunkNum - 1; j++) {
                CopyIn(offset, j, chunkSize);
                xChunk = Q_x.DeQue<T>();
                ReduceSum<T>(xChunk, xChunk, workspace, chunkSize);
                meanVal += (float)xChunk.GetValue(0);
                Q_x.FreeTensor(xChunk);
            }
            CopyIn(offset, (chunkNum - 1), lastChunkSize);
            xChunk = Q_x.DeQue<T>();
            ReduceSum<T>(xChunk, xChunk, workspace, lastChunkSize);
            meanVal += (float)xChunk.GetValue(0);
            meanVal /= groupSize;
            Q_x.FreeTensor(xChunk);
            // printf("[compute %d] mean: %f\n", i, meanVal);
            
            float varVal = 0.0f;
            for (int32_t j = 0; j < chunkNum - 1; j++) {
                CopyIn(offset, j, chunkSize);
                xChunk = Q_x.DeQue<T>();
                Adds(xChunk, xChunk, (T)(-meanVal), chunkSize);
                Mul(xChunk, xChunk, xChunk, chunkSize);
                ReduceSum<T>(xChunk, xChunk, workspace, chunkSize);
                varVal += (float)xChunk.GetValue(0);
                Q_x.FreeTensor(xChunk);
            }
            CopyIn(offset, (chunkNum - 1), lastChunkSize);
            xChunk = Q_x.DeQue<T>();
            Adds(xChunk, xChunk, (T)(-meanVal), lastChunkSize);
            Mul(xChunk, xChunk, xChunk, lastChunkSize);
            ReduceSum<T>(xChunk, xChunk, workspace, lastChunkSize);
            varVal += (float)xChunk.GetValue(0);
            varVal /= groupSize;
            Q_x.FreeTensor(xChunk);
            // printf("[compute %d] val: %f\n", i, varVal);

            LocalTensor<T> yChunk;
            float deno = (float)(1.0) / sqrt(varVal + eps);
            for (int32_t j = 0; j < chunkNum - 1; j++) {
                CopyIn(offset, j, chunkSize);
                xChunk = Q_x.DeQue<T>();
                yChunk = Q_y.AllocTensor<T>();
                Adds(xChunk, xChunk, (T)(-meanVal), chunkSize);
                Muls(yChunk, xChunk, (T)deno, chunkSize);
                Q_y.EnQue<T>(yChunk);
                CopyOut(offset, j, chunkSize);
                Q_x.FreeTensor(xChunk);
            }
            CopyIn(offset, (chunkNum - 1), lastChunkSize);
            xChunk = Q_x.DeQue<T>();
            yChunk = Q_y.AllocTensor<T>();
            Adds(xChunk, xChunk, (T)(-meanVal), lastChunkSize);
            Muls(yChunk, xChunk, (T)deno, lastChunkSize);
            Q_y.EnQue<T>(yChunk);
            CopyOut(offset, (chunkNum - 1), lastChunkSize);
            Q_x.FreeTensor(xChunk);
        }
    }

private:
    __aicore__ inline void InitTiling(GroupNormV2TilingData& tiling) {
        this->batchSize = tiling.batchSize;
        this->groupSize = tiling.groupSize;
        this->chunkSize =  tiling.chunkSize;
        this->chunkNum = tiling.chunkNum;
        this->lastChunkSize = tiling.lastChunkSize;
        this->eps = tiling.eps;
    }
    __aicore__ inline void CopyIn(int32_t offset, int chunkInx, int32_t length) {
        LocalTensor<T> xChunk = Q_x.AllocTensor<T>();
        DataCopy<T>(xChunk, Gm_x[offset + chunkInx * chunkSize], length);
        Q_x.EnQue(xChunk);
    }

    __aicore__ inline void CopyOut(int32_t offset, int chunkInx, int32_t length) {
        LocalTensor<T> yChunk = Q_y.DeQue<T>();
        DataCopy(Gm_y[offset + chunkInx * chunkSize], yChunk, length);
        Q_y.FreeTensor(yChunk);
    }

    __aicore__ inline void PrintParam() {
        printf("batchsize: %d\n", batchSize);
        printf("groupSize: %d\n", groupSize);
        printf("chunkSize: %d\n", chunkSize);
        printf("chunkNum: %d\n", chunkNum);
        printf("lastChunkSize: %d\n", lastChunkSize);
        printf("eps: %f\n", eps);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC>  B_work;

    GlobalTensor<T> Gm_x, Gm_y;

    // Calc by tiling.
    int32_t batchSize;
    int32_t groupSize;
    int32_t chunkSize;
    int32_t chunkNum;
    int32_t lastChunkSize;
    float eps;
};

class KernelSplitGroupNormV2 {
public:
    __aicore__ inline KernelSplitGroupNormV2() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR gamma,
        GM_ADDR beta,
        GM_ADDR y,
        GM_ADDR mean,
        GM_ADDR rstd,
        GroupNormV2TilingData& tiling
    ) {
        // Init global memory.
        InitTiling(tiling);
        int32_t totalDataSize = batchSize * groupSize;
        Gm_x.SetGlobalBuffer((__gm__ float *)x, totalDataSize);
        Gm_y.SetGlobalBuffer((__gm__ float *)y, totalDataSize);
        Gm_mean.SetGlobalBuffer((__gm__ float *)mean, totalChunkNum);
        Gm_var.SetGlobalBuffer((__gm__ float *)rstd, totalChunkNum);

        // Init buffer.
        pipe.InitBuffer(Q_x, 2, chunkSize * sizeof(float));
        pipe.InitBuffer(Q_inMean, BUFFER_NUM, totalChunkNum * sizeof(float));
        pipe.InitBuffer(Q_inVar, BUFFER_NUM, totalChunkNum * sizeof(float));
        pipe.InitBuffer(Q_y, 2, chunkSize * sizeof(float));
        pipe.InitBuffer(Q_outMean, BUFFER_NUM, chunkNum * sizeof(float));
        pipe.InitBuffer(Q_outVar, BUFFER_NUM, chunkNum * sizeof(float));
        pipe.InitBuffer(B_tmp, chunkSize);
        // PrintParam();
    }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < batchSize; i++) {
            int32_t batchOffset = i * groupSize;
            int32_t offset;
            // calc mean.
            LocalTensor<float> outMean = Q_outMean.AllocTensor<float>();
            LocalTensor<float> tmp = B_tmp.Get<float>();
            for (int j = 0; j < chunkNum - 1; j++) {
                offset = batchOffset + GetBlockIdx() * blockSize + j * chunkSize;
                CopyInput(offset, chunkSize);
                ComputeMean(outMean, j, chunkSize);
            }
            offset = batchOffset + GetBlockIdx() * blockSize + (chunkNum - 1) * chunkSize;
            CopyInput(offset, lastChunkSize);
            ComputeMean(outMean, chunkNum - 1, chunkSize);
            Q_outMean.EnQue<float>(outMean);
            offset = GetBlockIdx() * chunkNum;
            CopyOutMean(offset, chunkNum);
            SyncAll();
            GatherMean(0, totalChunkNum);
            LocalTensor<float> inMean = Q_inMean.DeQue<float>();
            Muls(inMean, inMean, (float)1 / groupSize, totalChunkNum);
            ReduceSum(inMean, inMean, tmp, totalChunkNum);
            
            // calc var.
            float meanVal = inMean.GetValue(0);
            LocalTensor<float> outVar = Q_outVar.AllocTensor<float>();
            for (int j = 0; j < chunkNum - 1; j++) {
                offset = batchOffset + GetBlockIdx() * blockSize + j * chunkSize;
                CopyInput(offset, chunkSize);
                ComputeVar(outVar, j, meanVal, chunkSize);
            }
            offset = batchOffset + GetBlockIdx() * blockSize + (chunkNum - 1) * chunkSize;
            CopyInput(offset, lastChunkSize);
            ComputeVar(outVar, chunkNum - 1, meanVal, lastChunkSize);
            Q_outVar.EnQue<float>(outVar);
            offset = GetBlockIdx() * chunkNum;
            CopyOutVar(offset, chunkNum);
            SyncAll();
            GatherVar(0, totalChunkNum);
            LocalTensor<float> inVar = Q_inVar.DeQue<float>();
            Muls(inVar, inVar, (float)1 / groupSize, totalChunkNum);
            ReduceSum(inVar, inVar, tmp, totalChunkNum);
            
            // calc norm
            Adds(inVar, inVar, eps, 1);
            Ln(inVar, inVar, 1);
            Muls(inVar, inVar, float(0.5), 1);
            Exp(inVar, inVar, 1);
            float deno = inVar.GetValue(0);
            // float deno = sqrt(inVar.GetValue(0) + eps);
            for (int j = 0; j < chunkNum - 1; j++) {
                offset = batchOffset + GetBlockIdx() * blockSize + j * chunkSize;
                CopyInput(offset, chunkSize);
                ComputeNorm(meanVal, deno, chunkSize);
                CopyOut(offset, chunkSize);
            }
            offset = batchOffset + GetBlockIdx() * blockSize + (chunkNum - 1) * chunkSize;
            CopyInput(offset, lastChunkSize);
            ComputeNorm(meanVal, deno, lastChunkSize);
            CopyOut(offset, lastChunkSize);

            Q_inMean.FreeTensor(inMean);
            Q_inVar.FreeTensor(inVar);
        }
    }

private:
    __aicore__ inline void InitTiling(GroupNormV2TilingData& tiling) {
        this->batchSize = tiling.batchSize;
        this->groupSize = tiling.groupSize;
        this->chunkSize =  tiling.chunkSize;
        this->chunkNum = tiling.chunkNum;
        this->lastChunkSize = tiling.lastChunkSize;
        this->eps = tiling.eps;
        this->blockSize = (chunkNum - 1) * chunkSize + lastChunkSize;
        this->totalChunkNum = GetBlockNum() * chunkNum;
    }
    __aicore__ inline void CopyInput(int32_t offset, int32_t length) {
        LocalTensor<float> xChunk = Q_x.AllocTensor<float>();
        DataCopy(xChunk, Gm_x[offset], length);
        Q_x.EnQue(xChunk);
    }
    __aicore__ inline void ComputeMean(LocalTensor<float>& mean, int32_t offset, int32_t length) {
        LocalTensor<float> x = Q_x.DeQue<float>();
        LocalTensor<float> tmp = B_tmp.Get<float>();
        ReduceSum(mean[offset], x, tmp, length);
        Q_x.FreeTensor<float>(x);
    }
    __aicore__ inline void CopyOutMean(int32_t offset, int32_t length) {
        LocalTensor<float> mean = Q_outMean.DeQue<float>();
        DataCopy(Gm_mean[offset], mean, length);
        Q_outMean.FreeTensor(mean);
    }
    __aicore__ inline void GatherMean(int32_t offset, int32_t length) {
        LocalTensor<float> mean = Q_inMean.AllocTensor<float>();
        DataCopy(mean, Gm_mean, length);
        Q_inMean.EnQue(mean);
    }
    __aicore__ inline void ComputeVar(
        LocalTensor<float> var,
        int32_t offset,
        const float& meanVal,
        int32_t length
    ) {
        LocalTensor<float> x = Q_x.DeQue<float>();
        LocalTensor<float> tmp = B_tmp.Get<float>();
        Adds(x, x, float(-meanVal), length);
        Mul(x, x, x, length);
        ReduceSum(var[offset], x, tmp, length);
        Q_x.FreeTensor<float>(x);
    }
    __aicore__ inline void CopyOutVar(int32_t offset, int32_t length) {
        LocalTensor<float> var = Q_outVar.DeQue<float>();
        DataCopy(Gm_var[offset], var, length);
        Q_outVar.FreeTensor(var);
    }
    __aicore__ inline void GatherVar(int32_t offset, int32_t length) {
        LocalTensor<float> var = Q_inVar.AllocTensor<float>();
        DataCopy(var, Gm_var, length);
        Q_inVar.EnQue(var);
    }
    __aicore__ inline void ComputeNorm(float meanVal, float deno, int32_t length) {
        LocalTensor<float> x = Q_x.DeQue<float>();
        LocalTensor<float> y = Q_y.AllocTensor<float>();
        Adds(x, x, float(-meanVal), length);
        Muls(y, x, float(1 / deno), length);
        Q_x.FreeTensor<float>(x);
        Q_y.EnQue<float>(y);
    }
    __aicore__ inline void CopyOut(int32_t offset, int32_t length) {
        LocalTensor<float> y = Q_y.DeQue<float>();
        DataCopy(Gm_y[offset], y, length);
        Q_y.FreeTensor(y);
    }

    __aicore__ inline void PrintParam() {
        printf("batchsize: %d\n", batchSize);
        printf("groupSize: %d\n", groupSize);
        printf("chunkSize: %d\n", chunkSize);
        printf("chunkNum: %d\n", chunkNum);
        printf("lastChunkSize: %d\n", lastChunkSize);
        printf("blockSize: %d\n", blockSize);
        printf("totalChunkNum: %d\n", totalChunkNum);
        printf("eps: %f\n", eps);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> Q_x;
    TQue<QuePosition::VECOUT, 2> Q_y;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_inMean, Q_inVar;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_outMean, Q_outVar;
    TBuf<QuePosition::VECCALC>  B_tmp;

    GlobalTensor<float> Gm_x, Gm_y, Gm_mean, Gm_var;

    // Calc by tiling.
    int32_t batchSize;
    int32_t groupSize;
    int32_t chunkSize;
    int32_t chunkNum;
    int32_t lastChunkSize;
    int32_t blockSize;
    int32_t totalChunkNum;
    float eps;
};

class KernelGroupNormV2FP16 {
public:
    __aicore__ inline KernelGroupNormV2FP16() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR gamma,
        GM_ADDR beta,
        GM_ADDR y,
        GM_ADDR mean,
        GM_ADDR rstd,
        GroupNormV2TilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);
        
        int32_t blockSize = batchSize * groupSize;
        xGm.SetGlobalBuffer((__gm__ half *)x + GetBlockIdx() * blockSize, blockSize);
        yGm.SetGlobalBuffer((__gm__ half*)y + GetBlockIdx() * blockSize, blockSize);
        
        pipe.InitBuffer(meanBuf, batchSize * sizeof(float));
        pipe.InitBuffer(rstdBuf, batchSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> meanLocal = meanBuf.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        for (int32_t i = 0; i < batchSize; ++i) {
            float sum = 0.0;
            for (int32_t k = 0; k < groupSize; ++k) {
                float val = (float)xGm.GetValue(i * groupSize + k);
                sum += val;
            }
            float avg = sum / groupSize;
            meanLocal.SetValue(i, (float)avg);
        }

        for (int32_t i = 0; i < batchSize; ++i) {
            float avg = (float)meanLocal.GetValue(i);
            float sum = 0.0;
            for (int32_t k = 0; k < groupSize; ++k) {
                float val = (float)xGm.GetValue(i * groupSize + k);
                sum += (val - avg) * (val - avg);
            }
            float var = sum / groupSize;
            rstdLocal.SetValue(i, (float)var);
        }

        for (int32_t i = 0; i < batchSize; ++i) {
            float mean = (float)meanLocal.GetValue(i);
            float variance = (float)rstdLocal.GetValue(i);
            float gamma = 1;
            float beta = 0;
            for (int32_t k = 0; k < groupSize; ++k) {
                float x = (float)xGm.GetValue(i * groupSize + k);
                float result = gamma * ((x - mean) / sqrt(variance + eps)) + beta;
                yGm.SetValue(i * groupSize + k, (half)result);
            }
        }
    }

private:
    __aicore__ inline void InitTiling(GroupNormV2TilingData& tiling) {
        this->batchSize = tiling.batchSize;
        this->groupSize = tiling.groupSize;
        this->chunkSize =  tiling.chunkSize;
        this->chunkNum = tiling.chunkNum;
        this->lastChunkSize = tiling.lastChunkSize;
        this->eps = tiling.eps;
    }

private:
    TPipe pipe;
    GlobalTensor<half> xGm, yGm;

    TBuf<TPosition::VECCALC> meanBuf, rstdBuf;

    int32_t batchSize, groupSize;
    int32_t chunkSize, chunkNum, lastChunkSize;
    float eps;
};


extern "C" __global__ __aicore__ void group_norm_v2(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR work = AscendC::GetUserWorkspace(workspace);
    GM_ADDR tmpMean = work;
    // GM_ADDR tmpVar = work + (GetBlockNum() * 8 * sizeof(float));
    // if (TILING_KEY_IS(1)) {
    //     KernelSplitGroupNormV2<float> op;
    //     op.Init(x, gamma, beta, y, work, work, tiling_data);
    //     op.Process();
    // }
    // else {
    //      KernelGroupNormV2FP16 op;
    //     op.Init(x, gamma, beta, y, mean, rstd, tiling_data);
    //     op.Process();
    // }

    if (sizeof(DTYPE_X) == 4) {
        if (tiling_data.groupSize > 1024) {
            KernelSplitGroupNormV2 op;
            op.Init(x, gamma, beta, y, work, work, tiling_data);
            op.Process();
        }
        else {
            KernelGroupNormV2<float> op;
            op.Init(x, gamma, beta, y, mean, rstd, tiling_data);
            op.Process();
        }
    }
    else {
        KernelGroupNormV2FP16 op;
        op.Init(x, gamma, beta, y, mean, rstd, tiling_data);
        op.Process();
    }
}