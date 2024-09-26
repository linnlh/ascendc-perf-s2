#include <cstring>
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template<typename T>
class KernelDepthToSpaceNHWC {
public:
    __aicore__ inline KernelDepthToSpaceNHWC() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        DepthToSpaceTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);
        int32_t dataSize = N * H * W * C;
        xGm.SetGlobalBuffer((__gm__ T *)x + GetBlockIdx() * dataSize, dataSize);
        yGm.SetGlobalBuffer((__gm__ T *)y + GetBlockIdx() * dataSize, dataSize);
    }

    __aicore__ inline void Process() {
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        int input_idx = n * hwcSize + h * wcSize + w * C + c;
                        int cur_h = h * blockSize + c / outC / blockSize;
                        int cur_w = w * blockSize + c / outC % blockSize;
                        int cur_c = c % outC;
                        int output_idx = n * outHwcSize + cur_h * outWcSize + cur_w * outC + cur_c;
                        yGm.SetValue(output_idx, xGm.GetValue(input_idx));
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void InitTiling(DepthToSpaceTilingData& tiling) {
        this->blockSize = tiling.blockSize;
        this->N = tiling.N;
        this->H = tiling.H;
        this->W = tiling.W;
        this->C = tiling.C;
        this->outN = this->N;
        this->outH = this->H * this->blockSize;
        this->outW = this->W * this->blockSize;
        this->outC = this->C / this->blockSize / this->blockSize;
        this->hwcSize = H * W * C;
        this->wcSize = W * C;
        this->outHwcSize = outH * outW * outC;
        this->outWcSize = outW * outC;
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    int32_t blockSize, N, H, W, C;
    int32_t outN, outH, outW, outC;

    int32_t hwcSize, wcSize;
    int32_t outHwcSize, outWcSize;
};

template<>
class KernelDepthToSpaceNHWC<float> {
    int32_t ALIGN = 32;
public:
    __aicore__ inline KernelDepthToSpaceNHWC() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        DepthToSpaceTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);
        int32_t dataSize = batchSize * W * C;
        Gm_x.SetGlobalBuffer((__gm__ float *)x + GetBlockIdx() * dataSize, dataSize);
        Gm_y.SetGlobalBuffer((__gm__ float *)y + GetBlockIdx() * dataSize, dataSize);
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < batchSize; i++) {
            int32_t offset = i * W * C;
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < blockSize; k++) {
                    int32_t inputOffset = offset + (j * blockSize + k) * strideSize;
                    int32_t outputOffset = offset + (k * W + j) * strideSize;
                    for (int n = 0; n < strideSize; n++) {
                        Gm_y.SetValue(outputOffset + n, Gm_x.GetValue(inputOffset + n));
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void InitTiling(DepthToSpaceTilingData& tiling) {
        this->blockSize = tiling.blockSize;
        this->N = tiling.N;
        this->H = tiling.H;
        this->W = tiling.W;
        this->C = tiling.C;
        this->outN = this->N;
        this->outH = this->H * this->blockSize;
        this->outW = this->W * this->blockSize;
        this->outC = this->C / this->blockSize / this->blockSize;
        this->batchSize = tiling.batchSize;
        this->strideSize = blockSize * outC;
        this->strideLine = this->strideSize / 8;
        if (this->strideSize % 8 != 0) {
            this->strideLine += 1;
        }
    }

private:

    GlobalTensor<float> Gm_x;
    GlobalTensor<float> Gm_y;

    int32_t N, H, W, C;
    int32_t outN, outH, outW, outC;
    int32_t blockSize;

    int32_t batchSize;
    int32_t strideSize;
    int32_t strideLine;
};

template<>
class KernelDepthToSpaceNHWC<half> {
    int32_t ALIGN = 32;
public:
    __aicore__ inline KernelDepthToSpaceNHWC() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        DepthToSpaceTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);
        int32_t dataSize = batchSize * W * C;
        Gm_x.SetGlobalBuffer((__gm__ half *)x + GetBlockIdx() * dataSize, dataSize);
        Gm_y.SetGlobalBuffer((__gm__ half *)y + GetBlockIdx() * dataSize, dataSize);

        pipe.InitBuffer(Q_x, BUFFER_NUM, W * C / strideSize * ALIGN);
        pipe.InitBuffer(Q_y, BUFFER_NUM,  W * C / strideSize * ALIGN);
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < batchSize; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void InitTiling(DepthToSpaceTilingData& tiling) {
        this->blockSize = tiling.blockSize;
        this->N = tiling.N;
        this->H = tiling.H;
        this->W = tiling.W;
        this->C = tiling.C;
        this->outN = this->N;
        this->outH = this->H * this->blockSize;
        this->outW = this->W * this->blockSize;
        this->outC = this->C / this->blockSize / this->blockSize;
        this->batchSize = tiling.batchSize;
        this->strideSize = blockSize * outC;
    }
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<half> x = Q_x.AllocTensor<half>();
        DataCopyExtParams copyParam = {
            uint16_t(W * C / strideSize),
            uint32_t(strideSize * sizeof(half)),
            0,
            0,
            0
        };
        DataCopyPadExtParams<half> padParam = {false, 0, 0, 0};
        DataCopyPad(x, Gm_x[progress * W * C], copyParam, padParam);
        Q_x.EnQue<half>(x);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<half> x = Q_x.DeQue<half>();
        LocalTensor<half> y = Q_y.AllocTensor<half>();

        int32_t offset = progress * W * C;
        for (int j = 0; j < W; j++) {
            for (int k = 0; k < blockSize; k++) {
                int32_t inputOffset = (j * blockSize + k) * ALIGN / sizeof(half);
                int32_t outputOffset = (k * W + j) * ALIGN / sizeof(half);
                Copy(y[outputOffset], x[inputOffset], strideSize, 1, {1, 1, 1, 1});
            }
        }
        Q_y.EnQue<half>(y);
        Q_x.FreeTensor(x);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<half> y = Q_y.DeQue<half>();
        DataCopyExtParams copyParam = {uint16_t(W * C / strideSize), uint32_t(strideSize * sizeof(half)), 0, 0, 0};   
        DataCopyPad(Gm_y[progress * W * C], y, copyParam);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

    GlobalTensor<half> Gm_x;
    GlobalTensor<half> Gm_y;

    int32_t N, H, W, C;
    int32_t outN, outH, outW, outC;
    int32_t blockSize;

    int32_t batchSize;
    int32_t strideSize;
};


class KernelDepthToSpaceNCHW {
public:
    __aicore__ inline KernelDepthToSpaceNCHW() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        DepthToSpaceTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);
        int32_t dataSize = N * H * W * C;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + GetBlockIdx() * dataSize, dataSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + GetBlockIdx() * dataSize, dataSize);
    }

    __aicore__ inline void Process() {
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int input_idx = n * C * H * W + c * H * W + h * W + w;
                        int cur_c = c % outC;
                        int cur_h = h * blockSize + c / outC / blockSize;
                        int cur_w = w * blockSize + c / outC % blockSize;
                        int output_idx = n * outC * outH * outW + cur_c * outH * outW + cur_h * outW + cur_w;
                        yGm.SetValue(output_idx, xGm.GetValue(input_idx));
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void InitTiling(DepthToSpaceTilingData& tiling) {
        this->blockSize = tiling.blockSize;
        this->N = tiling.N;
        this->H = tiling.H;
        this->W = tiling.W;
        this->C = tiling.C;
        this->outN = this->N;
        this->outH = this->H * this->blockSize;
        this->outW = this->W * this->blockSize;
        this->outC = this->C / this->blockSize / this->blockSize;
    }

private:

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t blockSize, N, H, W, C;
    int32_t outN, outH, outW, outC;
};

class KernelDepthToSpaceCRDNCHW {
public:
    __aicore__ inline KernelDepthToSpaceCRDNCHW() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        DepthToSpaceTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);
        int32_t dataSize = N * H * W * C;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + GetBlockIdx() * dataSize, dataSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + GetBlockIdx() * dataSize, dataSize);
    }

    __aicore__ inline void Process() {
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int input_idx = n * C * H * W + c * H * W + h * W + w;
                        int cur_c = c / (this->blockSize * this->blockSize);
                        int cur_h = h * blockSize + c % (this->blockSize * this->blockSize) / blockSize;
                        int cur_w = w * blockSize + c % (this->blockSize * this->blockSize) % blockSize;
                        int output_idx = n * outC * outH * outW + cur_c * outH * outW + cur_h * outW + cur_w;
                        yGm.SetValue(output_idx, xGm.GetValue(input_idx));
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void InitTiling(DepthToSpaceTilingData& tiling) {
        this->blockSize = tiling.blockSize;
        this->N = tiling.N;
        this->H = tiling.H;
        this->W = tiling.W;
        this->C = tiling.C;
        this->outN = this->N;
        this->outH = this->H * this->blockSize;
        this->outW = this->W * this->blockSize;
        this->outC = this->C / this->blockSize / this->blockSize;
    }

private:

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t blockSize, N, H, W, C;
    int32_t outN, outH, outW, outC;
};

class KernelDepthToSpaceCRDNHWC {
public:
    __aicore__ inline KernelDepthToSpaceCRDNHWC() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        DepthToSpaceTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);
        int32_t dataSize = N * H * W * C;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + GetBlockIdx() * dataSize, dataSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + GetBlockIdx() * dataSize, dataSize);
    }

    __aicore__ inline void Process() {
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        int input_idx = n * H * W * C + h * W * C + w * C + c;
                        int cur_h = h * blockSize + c % (this->blockSize * this->blockSize) / blockSize;
                        int cur_w = w * blockSize + c % (this->blockSize * this->blockSize) % blockSize;
                        int cur_c = c / (this->blockSize * this->blockSize);
                        int output_idx = n * outH * outW * outC + cur_h * outW * outC + cur_w * outC + cur_c;
                        yGm.SetValue(output_idx, xGm.GetValue(input_idx));
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void InitTiling(DepthToSpaceTilingData& tiling) {
        this->blockSize = tiling.blockSize;
        this->N = tiling.N;
        this->H = tiling.H;
        this->W = tiling.W;
        this->C = tiling.C;
        this->outN = this->N;
        this->outH = this->H * this->blockSize;
        this->outW = this->W * this->blockSize;
        this->outC = this->C / this->blockSize / this->blockSize;
    }

private:

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t blockSize, N, H, W, C;
    int32_t outN, outH, outW, outC;
};


extern "C" __global__ __aicore__ void depth_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.mode == 0) {
        KernelDepthToSpaceNHWC<DTYPE_X> op;
        op.Init(x, y, tiling_data);
        op.Process();
    }
    else if (tiling_data.mode == 1) {
        KernelDepthToSpaceCRDNHWC op;
        op.Init(x, y, tiling_data);
        op.Process();
    }
    else if (tiling_data.mode == 2) {
        KernelDepthToSpaceNCHW op;
        op.Init(x, y, tiling_data);
        op.Process();
    }
    else {
        KernelDepthToSpaceCRDNCHW op;
        op.Init(x, y, tiling_data);
        op.Process();
    }
}
