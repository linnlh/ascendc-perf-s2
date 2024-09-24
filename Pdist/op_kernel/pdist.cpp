#include "kernel_operator.h"
#include <cmath>

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template<typename T>
class KernelPdist;

template<>
class KernelPdist<float> {
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline void Init(
        TPipe& pipe,
        GM_ADDR x,
        GM_ADDR y,
        uint32_t N,
        uint32_t M,
        float p,
        uint32_t bigCoreNum,
        uint32_t bigCoreBatch,
        uint32_t smallCoreBatch
    ) {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t batchIdx = bigCoreBatch * coreIdx;
        this->coreBatch = bigCoreBatch;
        if (coreIdx >= bigCoreNum) {
            batchIdx -= (coreIdx - bigCoreNum) * (bigCoreBatch - smallCoreBatch);
            this->coreBatch = smallCoreBatch;
        }
        if (coreIdx == GetBlockNum() - 1) {
            this->coreBatch = N * (N - 1) / 2 - batchIdx;
        }
        this->batchIdx = batchIdx;
        // printf("batchIdx: %d, coreBatch: %d\n", batchIdx, coreBatch);
        row1 = -1;
        int32_t n = N - 1, cnt = 0;
        for (int32_t tmp = batchIdx; tmp >= 0;) {
            row1 += 1;
            cnt += n;
            tmp -= n;
            n -= 1;
        }
        row2 = (int32_t)batchIdx - (cnt - n - 1) + 1 + row1;
        // printf("row1: %d, row2: %d\n", row1, row2);

        Gm_x.SetGlobalBuffer((__gm__ float *)x, N * M);
        Gm_y.SetGlobalBuffer((__gm__ float *)y, N * (N - 1) / 2);

        this->N = N;
        this->M = M;
        this->p = p;
        // pipe.InitBuffer(B_temp, 8 * sizeof(float));
        pipe.InitBuffer(B_work, M * sizeof(float));
        pipe.InitBuffer(Q_x1, BUFFER_NUM, M * sizeof(float));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, M * sizeof(float));
        pipe.InitBuffer(Q_out, BUFFER_NUM, 8 * sizeof(float));
    }
    __aicore__ inline void Process() {
        // printf("row2: %d\n, N: %d\n", row2, N);
        uint32_t idx = 0;
        LocalTensor<float> work = B_work.Get<float>();
        CopyX1(row1);
        LocalTensor<float> x1 = Q_x1.DeQue<float>();
        for (; row2 < N; row2++) {
        //     LocalTensor<float> out = Q_out.AllocTensor<float>();
            // if (GetBlockIdx() == 0) {
            //     printf("idx: %d\n", idx);
            // }
            printf("");
            // PipeBarrier<PIPE_V>();
            float sum = 0;
            CopyX2(row2);
            LocalTensor<float> x2 = Q_x2.DeQue<float>();
            Sub(x2, x1, x2, M);
            Abs(x2, x2, M);
            Ln(x2, x2, M);
            Muls(x2, x2, p, M);
            Exp(x2, x2, M);
            ReduceSum(x2, x2, work, M);
            Ln(x2, x2, 1);
            Muls(x2, x2, (float)1.0 / p, 1);
            Exp(x2, x2, 1);
            Gm_y.SetValue(this->batchIdx + idx, x2.GetValue(0));
            idx += 1;
            Q_x2.FreeTensor(x2);
            if (idx >= coreBatch) {
                Q_x1.FreeTensor(x1);
                return;
            }
        }
        Q_x1.FreeTensor(x1);
        
        while (idx < coreBatch) {
            row1 = row1 + 1;
            // printf("block: %d, row1: %d\n", GetBlockIdx(), row1);
            for (; row1 < N - 1; row1++) {
                CopyX1(row1);
                LocalTensor<float> x1 = Q_x1.DeQue<float>();
                for (int row2 = row1 + 1; row2 < N; row2++) {
                    float sum = 0;
                    CopyX2(row2);
                    LocalTensor<float> x2 = Q_x2.DeQue<float>();
                    Sub(x2, x1, x2, M);
                    Abs(x2, x2, M);
                    Ln(x2, x2, M);
                    Muls(x2, x2, p, M);
                    Exp(x2, x2, M);
                    ReduceSum(x2, x2, work, M);
                    Ln(x2, x2, 1);
                    Muls(x2, x2, (float)1.0 / p, 1);
                    Exp(x2, x2, 1);
                    Gm_y.SetValue(this->batchIdx + idx, x2.GetValue(0));
                    idx += 1;
                    Q_x2.FreeTensor(x2);
                    if (idx >= coreBatch) {
                        Q_x1.FreeTensor(x1);
                        return;
                    }
                }
                Q_x1.FreeTensor(x1);
            }
        }
        
    }

private:
    __aicore__ inline void CopyX1(uint32_t row) {
        LocalTensor<float> x1 = Q_x1.AllocTensor<float>();
        DataCopy<float>(x1, Gm_x[row * M], M);
        Q_x1.EnQue<float>(x1);
    }

    __aicore__ inline void CopyX2(uint32_t row) {
        LocalTensor<float> x2 = Q_x2.AllocTensor<float>();
        DataCopy<float>(x2, Gm_x[row * M], M);
        Q_x2.EnQue<float>(x2);
    }

    __aicore__ inline void CopyOut(uint32_t idx) {
        LocalTensor<float> out = Q_out.DeQue<float>();
        // printf("idx: %d, out: %f\n", idx, out.GetValue(0));
        DataCopy<float>(Gm_y[idx], out, 1);
        Q_out.FreeTensor(out);
    }
private:
    GlobalTensor<float> Gm_x, Gm_y;
    
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_out;
    // TBuf<QuePosition::VECCALC> B_temp, B_max;
    TBuf<QuePosition::VECCALC> B_work;

    uint32_t coreBatch;
    uint32_t N, M;
    float p;
    uint32_t row1, row2, batchIdx;
};

template<>
class KernelPdist<half> {
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline void Init(
        TPipe& pipe,
        GM_ADDR x,
        GM_ADDR y,
        uint32_t N,
        uint32_t M,
        float p,
        uint32_t bigCoreNum,
        uint32_t bigCoreBatch,
        uint32_t smallCoreBatch
    ) {
        Gm_x.SetGlobalBuffer((__gm__ half *)x, N * M);
        Gm_y.SetGlobalBuffer((__gm__ half *)y, N * (N - 1) / 2);

        this->N = N;
        this->M = M;
        this->p = p;
        this->coreBatch = N * (N - 1) / 2;
        pipe.InitBuffer(B_temp, 8 * sizeof(float));
    }
    __aicore__ inline void Process() {
        LocalTensor<float> temp = B_temp.Get<float>();
        uint32_t idx = 0;
        for (int row1 = 0; row1 < N - 1; row1++) {
            for (int row2 = row1 + 1; row2 < N; row2++) {
                float sum = 0;
                float temp_max = 0;
                for (int k = 0; k < M; k++) {
                    float val1 = (float)Gm_x.GetValue(row1 * M + k);
                    float val2 = (float)Gm_x.GetValue(row2 * M + k);
                    float diff = std::abs(val1 - val2);
                    if (diff > temp_max) {
                        temp_max = diff;
                    }
                }

                for (int k = 0; k < M; k++) {
                    float val1 = (float)Gm_x.GetValue(row1 * M + k);
                    float val2 = (float)Gm_x.GetValue(row2 * M + k);
                    temp.SetValue(0, (float)std::abs(val1 - val2) / temp_max);
                    Ln(temp, temp, 1);
                    Muls(temp, temp, p, 1);
                    Exp(temp, temp, 1);
                    sum += temp.GetValue(0);
                }
                temp.SetValue(0, (float)(sum));
                Ln(temp, temp, 1);
                Muls(temp, temp, ((float)1.0 / p), 1);
                Exp(temp, temp, 1);
                Muls(temp, temp, temp_max, 1);
                Gm_y.SetValue(idx, (half)temp.GetValue(0));
                idx += 1;
            }
        }
    }
private:
    GlobalTensor<half> Gm_x, Gm_y;

    TBuf<QuePosition::VECCALC> B_temp;

    uint32_t coreBatch;
    uint32_t N, M;
    float p;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelPdist<DTYPE_X> op;
    TPipe pipe;
    op.Init(pipe, x, y, tiling_data.N, tiling_data.M, tiling_data.p, tiling_data.bigCoreNum, tiling_data.bigCoreBatch, tiling_data.smallCoreBatch);
    op.Process();

    // KernelPdist<float> op;
    // TPipe pipe;
    // op.Init(pipe, x, y, tiling_data.N, tiling_data.M, tiling_data.p, tiling_data.bigCoreNum, tiling_data.bigCoreBatch, tiling_data.smallCoreBatch);
    // op.Process();
}