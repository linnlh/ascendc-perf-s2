
#include <cstring>
#include <cmath>
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;


template<typename T>
class KernelBallQuery {
    int32_t ALIGN = 32;
    int32_t dataBlockLength = ALIGN / sizeof(T);
public:
    __aicore__ inline KernelBallQuery() {}
        __aicore__ inline void Init(
        GM_ADDR xyz,
        GM_ADDR center_xyz,
        GM_ADDR outputIdx,
        BallQueryTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);

        // Set global memory.
        int64_t xyzLength = batchSize * xyzNum * 3;
        int64_t centerLength = batchSize * centerNum * 3;
        int64_t indexLength = batchSize * centerNum * numSample;
        xyzGm.SetGlobalBuffer((__gm__ T *)xyz + GetBlockIdx() * xyzLength, xyzLength);
        centerxyzGm.SetGlobalBuffer((__gm__ T *)center_xyz + GetBlockIdx() * centerLength, centerLength);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)outputIdx + GetBlockIdx() * indexLength, indexLength);

        // Set local memory.
        pipe.InitBuffer(inQueueXYZ, BUFFER_NUM, xyzNum * ALIGN);
        pipe.InitBuffer(inQueueCenter, BUFFER_NUM, centerNum * ALIGN);
        pipe.InitBuffer(outQueueIndex, BUFFER_NUM, centerNum * numSample * sizeof(int32_t));
        pipe.InitBuffer(tempCenterBuf, ALIGN);
        pipe.InitBuffer(B_dist, xyzNum * ALIGN);
        // PrintParam();
    }
    __aicore__ inline void Process() {
        int32_t loopRound = this->batchSize;
        for (int i = 0; i < loopRound; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void InitTiling(BallQueryTilingData& tiling) {
        this->batchSize = tiling.batchSize;
        this->xyzNum = tiling.xyzNum;
        this->centerNum = tiling.centerNum;
        this->numSample = tiling.numSample;
        this->minRadius2 = tiling.minRadius2;
        this->maxRadius2 = tiling.maxRadius2;
    }
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<T> xyzLocal = inQueueXYZ.AllocTensor<T>();
        LocalTensor<T> centerLocal = inQueueCenter.AllocTensor<T>();
        DataCopyExtParams copyParams1 {(uint16_t)xyzNum, 3 * sizeof(T), 0, 0, 0};
        DataCopyExtParams copyParams2 {(uint16_t)centerNum, 3 * sizeof(T), 0, 0, 0};
        uint8_t padding = (uint8_t)(dataBlockLength - 3);
        DataCopyPadExtParams<T> padParams {true, 0, padding, 0}; 
        DataCopyPad(xyzLocal, xyzGm[progress * xyzNum * 3], copyParams1, padParams);
        DataCopyPad(centerLocal, centerxyzGm[progress * centerNum * 3], copyParams2, padParams);
        inQueueXYZ.EnQue<T>(xyzLocal);
        inQueueCenter.EnQue<T>(centerLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<T> xyzLocal = inQueueXYZ.DeQue<T>();
        LocalTensor<T> centerLocal = inQueueCenter.DeQue<T>();
        LocalTensor<T> dist = B_dist.Get<T>();
        LocalTensor<int32_t> indexLocal = outQueueIndex.AllocTensor<int32_t>();

        for (int i = 0; i < centerNum; i++) {
            Sub(dist, xyzLocal, centerLocal[dataBlockLength * i], dataBlockLength, xyzNum, {1, 1, 1, 1, 1, 0});
            Mul(dist, dist, dist, xyzNum * dataBlockLength);
            // WholeReduceSum(dist, dist, 3, xyzNum, dataBlockLength, 1, 1);
            // Sum(dist, dist, {(uint32_t)xyzNum, 8, 3});
            LocalTensor<int32_t> index = indexLocal[i * numSample];
            int cnt = 0;
            for (int j = 0; j < xyzNum; j++) {
                int32_t distOffset = j * dataBlockLength;
                // float dist2 = (float)dist.GetValue(j);
                float dist2 = (float)(dist.GetValue(distOffset)) + (float)(dist.GetValue(distOffset + 1)) + (float)(dist.GetValue(distOffset + 2));
                if (minRadius2 <= dist2 && dist2 < maxRadius2) {
                    if (cnt == 0) {
                        for (int k = 0; k < numSample; k++)
                            index.SetValue(k, j);
                    }
                    index.SetValue(cnt, j);
                    cnt += 1;
                    if (cnt >= numSample) {
                        break;
                    }
                }
            }
        }

        outQueueIndex.EnQue<int32_t>(indexLocal);
        inQueueXYZ.FreeTensor(xyzLocal);
        inQueueCenter.FreeTensor(centerLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<int32_t> indexLocal = outQueueIndex.DeQue<int32_t>();
        DataCopy(idxGm[progress * centerNum * numSample], indexLocal, centerNum * numSample);
        outQueueIndex.FreeTensor(indexLocal);
    }

    __aicore__ inline void PrintParam() {
        printf("=========Tiling=========\n");
        printf("batchSize: %ld\n", batchSize);
        printf("xyzNum: %ld\n", xyzNum);
        printf("centerNum: %ld\n", centerNum);
        printf("numSample: %ld\n", numSample);
        printf("minRadius2: %f\n", minRadius2);
        printf("maxRadius2: %f\n", maxRadius2);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueXYZ;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCenter;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndex;
    TBuf<QuePosition::VECCALC> tempCenterBuf, B_dist;

    GlobalTensor<T> xyzGm;
    GlobalTensor<T> centerxyzGm;
    GlobalTensor<int32_t> idxGm;

    // Tiling data
    int64_t batchSize;
    int64_t xyzNum;
    int64_t centerNum;
    int64_t numSample;
    float minRadius2;
    float maxRadius2;
};

// template<>
// class KernelBallQuery<float> {
//     int32_t ALIGN = 32;
//     int32_t dataBlockLength = ALIGN / sizeof(float);
// public:
//     __aicore__ inline KernelBallQuery() {}
//         __aicore__ inline void Init(
//         GM_ADDR xyz,
//         GM_ADDR center_xyz,
//         GM_ADDR outputIdx,
//         BallQueryTilingData& tiling
//     ) {
//         ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
//         InitTiling(tiling);

//         // Set global memory.
//         int64_t xyzLength = batchSize * xyzNum * 3;
//         int64_t centerLength = batchSize * centerNum * 3;
//         int64_t indexLength = batchSize * centerNum * numSample;
//         xyzGm.SetGlobalBuffer((__gm__ float *)xyz + GetBlockIdx() * xyzLength, xyzLength);
//         centerxyzGm.SetGlobalBuffer((__gm__ float *)center_xyz + GetBlockIdx() * centerLength, centerLength);
//         idxGm.SetGlobalBuffer((__gm__ int32_t *)outputIdx + GetBlockIdx() * indexLength, indexLength);

//         // Set local memory.
//         pipe.InitBuffer(inQueueXYZ, BUFFER_NUM, xyzNum * ALIGN);
//         pipe.InitBuffer(inQueueCenter, BUFFER_NUM, centerNum * ALIGN);
//         pipe.InitBuffer(outQueueIndex, BUFFER_NUM, centerNum * numSample * sizeof(int32_t));
//         pipe.InitBuffer(B_dist, xyzNum * ALIGN);
//         // PrintParam();
//     }
//     __aicore__ inline void Process() {
//         int32_t loopRound = this->batchSize;
//         for (int i = 0; i < loopRound; i++) {
//             CopyIn(i);
//             Compute(i);
//             CopyOut(i);
//         }
//     }

// private:
//     __aicore__ inline void InitTiling(BallQueryTilingData& tiling) {
//         this->batchSize = tiling.batchSize;
//         this->xyzNum = tiling.xyzNum;
//         this->centerNum = tiling.centerNum;
//         this->numSample = tiling.numSample;
//         this->minRadius2 = tiling.minRadius2;
//         this->maxRadius2 = tiling.maxRadius2;
//         repeatTime = xyzNum / 64;
//         lastRepeatChunk = xyzNum % 64;
//     }
//     __aicore__ inline void CopyIn(int32_t progress) {
//         LocalTensor<float> xyzLocal = inQueueXYZ.AllocTensor<float>();
//         LocalTensor<float> centerLocal = inQueueCenter.AllocTensor<float>();
//         DataCopyExtParams copyParams1 {(uint16_t)xyzNum, 3 * sizeof(float), 0, 0, 0};
//         DataCopyExtParams copyParams2 {(uint16_t)centerNum, 3 * sizeof(float), 0, 0, 0};
//         uint8_t padding = (uint8_t)(dataBlockLength - 3);
//         DataCopyPadExtParams<float> padParams {true, 0, padding, 0}; 
//         DataCopyPad(xyzLocal, xyzGm[progress * xyzNum * 3], copyParams1, padParams);
//         DataCopyPad(centerLocal, centerxyzGm[progress * centerNum * 3], copyParams2, padParams);
//         inQueueXYZ.EnQue<float>(xyzLocal);
//         inQueueCenter.EnQue<float>(centerLocal);
//     }
//     __aicore__ inline void Compute(int32_t progress) {
//         LocalTensor<float> xyzLocal = inQueueXYZ.DeQue<float>();
//         LocalTensor<float> centerLocal = inQueueCenter.DeQue<float>();
//         LocalTensor<float> dist = B_dist.Get<float>();
//         LocalTensor<int32_t> indexLocal = outQueueIndex.AllocTensor<int32_t>();

//         for (int i = 0; i < centerNum; i++) {
//             int cnt = 0;
//             for (int k = 0; k < repeatTime; k++) {
//                 Sub(dist[64 * k], xyzLocal[64 * k * 8], centerLocal[8 * i], 3, 64, {1, 1, 1, 1, 1, 0});
//                 Mul
//             }


//             // for (int k = 0; k < repeatTime; k++)
//             //     Sub(dist[k * 255 * 8], xyzLocal[k * 255 * 8], centerLocal[8 * i], 8, 255, {1, 1, 1, 1, 1, 0});
//             // if (lastRepeatChunk)
//             //     Sub(dist[repeatTime * 255 * 8], xyzLocal[repeatTime * 255 * 8], centerLocal[8 * i], 8, lastRepeatChunk, {1, 1, 1, 1, 1, 0});
//             // Mul(dist, dist, dist, xyzNum * dataBlockLength);
//             // // WholeReduceSum(dist, dist, 3, xyzNum, dataBlockLength, 1, 1);
//             // Sum(dist, dist, {(uint32_t)xyzNum, 8, 3});
//             // LocalTensor<int32_t> index = indexLocal[i * numSample];
//             // int cnt = 0;
//             // for (int j = 0; j < xyzNum; j++) {
//             //     int32_t distOffset = j * dataBlockLength;
//             //     float dist2 = (float)dist.GetValue(j);
//             //     // float dist2 = (float)(dist.GetValue(distOffset)) + (float)(dist.GetValue(distOffset + 1)) + (float)(dist.GetValue(distOffset + 2));
//             //     if (minRadius2 <= dist2 && dist2 < maxRadius2) {
//             //         if (cnt == 0) {
//             //             for (int k = 0; k < numSample; k++)
//             //                 index.SetValue(k, j);
//             //         }
//             //         index.SetValue(cnt, j);
//             //         cnt += 1;
//             //         if (cnt >= numSample) {
//             //             break;
//             //         }
//             //     }
//             // }
//         }

//         outQueueIndex.EnQue<int32_t>(indexLocal);
//         inQueueXYZ.FreeTensor(xyzLocal);
//         inQueueCenter.FreeTensor(centerLocal);
//     }
//     __aicore__ inline void CopyOut(int32_t progress) {
//         LocalTensor<int32_t> indexLocal = outQueueIndex.DeQue<int32_t>();
//         DataCopy(idxGm[progress * centerNum * numSample], indexLocal, centerNum * numSample);
//         outQueueIndex.FreeTensor(indexLocal);
//     }

//     __aicore__ inline void PrintParam() {
//         printf("=========Tiling=========\n");
//         printf("batchSize: %ld\n", batchSize);
//         printf("xyzNum: %ld\n", xyzNum);
//         printf("centerNum: %ld\n", centerNum);
//         printf("numSample: %ld\n", numSample);
//         printf("minRadius2: %f\n", minRadius2);
//         printf("maxRadius2: %f\n", maxRadius2);
//     }

// private:
//     TPipe pipe;
//     TQue<QuePosition::VECIN, BUFFER_NUM> inQueueXYZ;
//     TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCenter;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndex;
//     TBuf<QuePosition::VECCALC> B_dist;

//     GlobalTensor<float> xyzGm;
//     GlobalTensor<float> centerxyzGm;
//     GlobalTensor<int32_t> idxGm;

//     // Tiling data
//     int64_t batchSize;
//     int64_t xyzNum;
//     int64_t centerNum;
//     int64_t numSample;
//     float minRadius2;
//     float maxRadius2;
//     int32_t repeatTime, lastRepeatChunk;
// };

// template<>
// class KernelBallQuery<float> {
//     int32_t ALIGN = 32;
//     int32_t dataBlockLength = ALIGN / sizeof(float);
//     const int32_t CHUNK_SIZE = 64;
// public:
//     __aicore__ inline KernelBallQuery() {}
//         __aicore__ inline void Init(
//         GM_ADDR xyz,
//         GM_ADDR center_xyz,
//         GM_ADDR outputIdx,
//         BallQueryTilingData& tiling
//     ) {
//         ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
//         InitTiling(tiling);

//         // Set global memory.
//         int64_t xyzLength = batchSize * xyzNum * 3;
//         int64_t centerLength = batchSize * centerNum * 3;
//         int64_t indexLength = batchSize * centerNum * numSample;
//         Gm_x.SetGlobalBuffer((__gm__ float *)xyz + GetBlockIdx() * xyzLength, xyzLength);
//         Gm_center.SetGlobalBuffer((__gm__ float *)center_xyz + GetBlockIdx() * centerLength, centerLength);
//         Gm_indice.SetGlobalBuffer((__gm__ int32_t *)outputIdx + GetBlockIdx() * indexLength, indexLength);

//         // Set local memory.
//         pipe.InitBuffer(Q_x, BUFFER_NUM, xyzNum * ALIGN);
//         pipe.InitBuffer(Q_center, BUFFER_NUM, ALIGN);
//         pipe.InitBuffer(Q_indice, BUFFER_NUM, numSample * sizeof(int32_t));
//         pipe.InitBuffer(B_dist, CHUNK_SIZE * ALIGN);
//         pipe.InitBuffer(B_mask1, CHUNK_SIZE);
//         pipe.InitBuffer(B_mask2, CHUNK_SIZE);
//         // PrintParam();
//     }
//     __aicore__ inline void Process() {
//         for (int i = 0; i < batchSize; i++) {
//             CopyIn(i);
//             LocalTensor<float> x = Q_x.DeQue<float>();
//             for (int j = 0; j < centerNum; j++) {
//                 CopyInCenter(i, j);
//                 Compute(x, i, j);
//                 CopyOut(i, j);
//             }
//             Q_x.FreeTensor(x);
//         }
//     }

// private:
//     __aicore__ inline void InitTiling(BallQueryTilingData& tiling) {
//         this->batchSize = tiling.batchSize;
//         this->xyzNum = tiling.xyzNum;
//         this->centerNum = tiling.centerNum;
//         this->numSample = tiling.numSample;
//         this->minRadius2 = tiling.minRadius2;
//         this->maxRadius2 = tiling.maxRadius2;
//         chunkNum = xyzNum / 64;
//         lastChunkSize = xyzNum % 64;
//     }
//     __aicore__ inline void CopyIn(int32_t batchIdx) {
//         int32_t offset = batchIdx * xyzNum * 3;
//         LocalTensor<float> x = Q_x.AllocTensor<float>();
//         DataCopyExtParams copyParam {(uint16_t)xyzNum, 3 * sizeof(float), 0, 0, 0};
//         DataCopyPadExtParams<float> padParam {false, 0, 0, 0};
//         DataCopyPad(x, Gm_x[offset], copyParam, padParam);
//         Q_x.EnQue<float>(x);
//     }
//     __aicore__ inline void CopyInCenter(int32_t batchIdx, int32_t centerIdx) {
//         int32_t offset = batchIdx * centerNum * 3 + centerIdx * 3;
//         LocalTensor<float> center = Q_center.AllocTensor<float>();
//         DataCopyExtParams copyParam {(uint16_t)1, 3 * sizeof(float), 0, 0, 0};
//         DataCopyPadExtParams<float> padParam {false, 0, 0, 0};
//         DataCopyPad(center, Gm_center[offset], copyParam, padParam);
//         Q_center.EnQue<float>(center);
//     }
//     __aicore__ inline void Compute(LocalTensor<float>& x, int32_t batchIdx, int32_t centerIdx) {
//         LocalTensor<float> center = Q_center.DeQue<float>();
//         LocalTensor<float> dist = B_dist.Get<float>();
//         LocalTensor<int32_t> indice = Q_indice.AllocTensor<int32_t>();
//         LocalTensor<float> mask1 = B_mask1.Get<float>();
//         LocalTensor<float> mask2 = B_mask2.Get<float>();

//         int32_t cnt = 0;
//         int32_t indexOffset = 0;
//         int32_t xOffset;
//         for (int i = 0; i < chunkNum; i++) {
//             if (cnt >= numSample) break;
//             indexOffset = i * CHUNK_SIZE;
//             xOffset = i * CHUNK_SIZE * 8;
//             Sub(dist, x[xOffset], center, 3, CHUNK_SIZE, {1, 1, 1, 1, 1, 0});
//             Mul(dist, dist, dist, 3, CHUNK_SIZE, {1, 1, 1, 1, 1, 1});
//             WholeReduceSum(dist, dist, 3, CHUNK_SIZE, 1, 1, 1);
//             CompareScalar(mask1, dist, minRadius2, CMPMODE::GE, 1, CHUNK_SIZE, {1, 1, 1, 1});
//             CompareScalar(mask2, dist, maxRadius2, CMPMODE::LT, 1, CHUNK_SIZE, {1, 1, 1, 1});
//             And(mask1, mask1, mask2, CHUNK_SIZE);
//             uint64_t maskVal = mask1.ReinterpretCast<uint64_t>().GetValue(0);
//             int64_t firstOneBit = ScalarGetSFFValue<1>(maskVal);
//             if (firstOneBit != -1) {
//                 int32_t index = indexOffset + firstOneBit;
//                 if (cnt == 0) {
//                     Duplicate(indice, index, numSample);
//                     maskVal >>= (firstOneBit + 1);
//                     indexOffset += (firstOneBit + 1);
//                     cnt += 1;
//                     firstOneBit = ScalarGetSFFValue<1>(maskVal);
//                 }
//                 while (firstOneBit != -1 && cnt < numSample) {
//                     index = indexOffset + firstOneBit;
//                     indice.SetValue(cnt, index);
//                     maskVal >>= (firstOneBit + 1);
//                     indexOffset += (firstOneBit + 1);
//                     cnt += 1;
//                     firstOneBit = ScalarGetSFFValue<1>(maskVal);
//                 }
//             }
//         }
//         if (cnt < numSample && lastChunkSize) {
//             indexOffset = chunkNum * CHUNK_SIZE;
//             xOffset = chunkNum * CHUNK_SIZE * 8;
//             Sub(dist, x[xOffset], center, 3, lastChunkSize, {1, 1, 1, 1, 1, 0});
//             Mul(dist, dist, dist, 3, lastChunkSize, {1, 1, 1, 1, 1, 1});
//             WholeReduceSum(dist, dist, 3, lastChunkSize, 1, 1, 1);
//             CompareScalar(mask1, dist, minRadius2, CMPMODE::GE, 1, lastChunkSize, {1, 1, 1, 1});
//             CompareScalar(mask2, dist, maxRadius2, CMPMODE::LT, 1, lastChunkSize, {1, 1, 1, 1});
//             And(mask1, mask1, mask2, lastChunkSize);
//             uint64_t maskVal = mask1.ReinterpretCast<uint64_t>().GetValue(0);
//             int64_t firstOneBit = ScalarGetSFFValue<1>(maskVal);
//             if (firstOneBit != -1) {
//                 int32_t index = indexOffset + firstOneBit;
//                 if (cnt == 0) {
//                     Duplicate(indice, index, numSample);
//                     maskVal >>= (firstOneBit + 1);
//                     indexOffset += (firstOneBit + 1);
//                     cnt += 1;
//                     firstOneBit = ScalarGetSFFValue<1>(maskVal);
//                 }
//                 while (firstOneBit != -1 && cnt < numSample) {
//                     index = indexOffset + firstOneBit;
//                     indice.SetValue(cnt, index);
//                     maskVal >>= (firstOneBit + 1);
//                     indexOffset += (firstOneBit + 1);
//                     cnt += 1;
//                     firstOneBit = ScalarGetSFFValue<1>(maskVal);
//                 }
//             }
//         }
//         Q_center.FreeTensor(center);
//         Q_indice.EnQue<int32_t>(indice);
//     }
//     __aicore__ inline void CopyOut(int32_t batchIdx, int32_t centerIdx) {
//         LocalTensor<int32_t> indice = Q_indice.DeQue<int32_t>();
//         int32_t offset = batchIdx * centerNum * numSample + centerIdx * numSample;
//         DataCopyExtParams copyParam {1, uint32_t(numSample * sizeof(int32_t)), 0, 0, 0};
//         DataCopyPad(Gm_indice[offset], indice, copyParam);
//         Q_indice.FreeTensor(indice);
//     }

//     __aicore__ inline void PrintParam() {
//         printf("=========Tiling=========\n");
//         printf("batchSize: %ld\n", batchSize);
//         printf("xyzNum: %ld\n", xyzNum);
//         printf("centerNum: %ld\n", centerNum);
//         printf("numSample: %ld\n", numSample);
//         printf("minRadius2: %f\n", minRadius2);
//         printf("maxRadius2: %f\n", maxRadius2);
//         printf("chunkNum: %d\n", chunkNum);
//         printf("lastChunkSize: %d\n", lastChunkSize);
//     }

// private:
//     TPipe pipe;
//     TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
//     TQue<QuePosition::VECIN, BUFFER_NUM> Q_center;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> Q_indice;
//     TBuf<> B_dist, B_mask1, B_mask2;

//     GlobalTensor<float> Gm_x, Gm_center;
//     GlobalTensor<int32_t> Gm_indice;

//     // Tiling data
//     int64_t batchSize;
//     int64_t xyzNum;
//     int64_t centerNum;
//     int64_t numSample;
//     float minRadius2, maxRadius2;
//     int32_t chunkNum, lastChunkSize;
// };


template<>
class KernelBallQuery<float> {
    const int32_t ALIGN = 32;
    const int32_t CHUNK_SIZE = 64;
public:
    __aicore__ inline KernelBallQuery() {}
        __aicore__ inline void Init(
        GM_ADDR xyz,
        GM_ADDR center_xyz,
        GM_ADDR outputIdx,
        BallQueryTilingData& tiling
    ) {
        InitTiling(tiling);

        Gm_x.SetGlobalBuffer((__gm__ float *)xyz, batchSize * xyzNum * 3);
        Gm_center.SetGlobalBuffer((__gm__ float *)center_xyz, batchSize * centerNum * 3);
        Gm_indice.SetGlobalBuffer((__gm__ int32_t *)outputIdx, batchSize * centerNum * numSample);

        pipe.InitBuffer(Q_center, BUFFER_NUM, blockCenterNum * ALIGN);
        pipe.InitBuffer(Q_x, BUFFER_NUM, xyzNum * ALIGN);
        pipe.InitBuffer(Q_indice, BUFFER_NUM, blockCenterNum * numSampleAlign * sizeof(int32_t));
        pipe.InitBuffer(B_dist, CHUNK_SIZE * ALIGN);
        pipe.InitBuffer(B_mask1, CHUNK_SIZE);
        pipe.InitBuffer(B_mask2, CHUNK_SIZE);
        // if (GetBlockIdx() == 0)
        //     PrintParam();
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < batchSize; i++) {
            CopyIn(i);
            CopyCenter(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void InitTiling(BallQueryTilingData& tiling) {
        this->batchSize = tiling.batchSize;
        this->xyzNum = tiling.xyzNum;
        this->centerNum = tiling.centerNum;
        this->numSample = tiling.numSample;
        this->minRadius2 = tiling.minRadius2;
        this->maxRadius2 = tiling.maxRadius2;
        chunkNum = xyzNum / 64;
        lastChunkSize = xyzNum % 64;
        blockCenterNum = xyzNum / GetBlockNum();
        numSampleAlign = numSample + tiling.indicePadding;
    }
    __aicore__ inline void CopyIn(int32_t progress) {
        int32_t offset = progress * xyzNum * 3;
        LocalTensor<float> x = Q_x.AllocTensor<float>();
        DataCopyExtParams copyParam = {(uint16_t)xyzNum, 3 * sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> padParam = {false, 0, 0, 0};
        DataCopyPad(x, Gm_x[offset], copyParam, padParam);
        Q_x.EnQue(x);
    }
    __aicore__ inline void CopyCenter(int32_t progress) {
        int32_t offset = progress * centerNum * 3 + GetBlockIdx() * blockCenterNum * 3;
        LocalTensor<float> center = Q_center.AllocTensor<float>();
        DataCopyExtParams copyParam = {(uint16_t)blockCenterNum, 3 * sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> padParam = {false, 0, 0, 0};
        DataCopyPad(center, Gm_center[offset], copyParam, padParam);
        Q_center.EnQue(center);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> x = Q_x.DeQue<float>();
        LocalTensor<float> center = Q_center.DeQue<float>();
        LocalTensor<float> dist = B_dist.Get<float>();
        LocalTensor<float> mask1 = B_mask1.Get<float>();
        LocalTensor<float> mask2 = B_mask2.Get<float>();
        LocalTensor<int32_t> indice = Q_indice.AllocTensor<int32_t>();
        for (int i = 0; i < blockCenterNum; i++) {
            int32_t cnt = 0;
            int32_t indiceOffset = i * numSampleAlign;
            int32_t indexOffset = 0;
            for (int j = 0; j < chunkNum; j++) {
                // printf("cnt: %d\n", cnt);
                // if (GetBlockIdx() == 0 && progress == 0 && i == 1) {
                //     printf("indice: ");
                //     for (int n = 0; n < numSample; n++) {
                //         printf("%d ", indice.GetValue(indiceOffset + n));
                //     }
                //     printf("\n\n");
                // }
                if (cnt >= numSample) break;
                indexOffset = j * CHUNK_SIZE;
                Sub(dist, x[j * CHUNK_SIZE * 8], center[i * 8], 3, CHUNK_SIZE, {1, 1, 1, 1, 1, 0});
                // if (GetBlockIdx() == 1 && progress == 0 && i == 1) {
                //     for (int n = 0; n < 8; n++) {
                //         printf("dist[%d]: %f, %f, %f\n", n, dist.GetValue(n * 8), dist.GetValue(n * 8 + 1), dist.GetValue(n * 8 + 2));
                //     }
                //     printf("\n\n");
                // }
                Mul(dist, dist, dist, 3, CHUNK_SIZE, {1, 1, 1, 1, 1, 1});
                WholeReduceSum(dist, dist, 3, CHUNK_SIZE, 1, 1, 1);
                CompareScalar(mask1, dist, minRadius2, CMPMODE::GE, 1, CHUNK_SIZE, {1, 1, 1, 1});
                CompareScalar(mask2, dist, maxRadius2, CMPMODE::LT, 1, CHUNK_SIZE, {1, 1, 1, 1});
                And(mask1, mask1, mask2, CHUNK_SIZE);
                uint64_t maskVal = mask1.ReinterpretCast<uint64_t>().GetValue(0);
                int64_t firstOneBit = ScalarGetSFFValue<1>(maskVal);
                if (firstOneBit != -1) {
                    int32_t index = indexOffset + firstOneBit;
                    if (cnt == 0) {
                        Duplicate(indice[indiceOffset], index, numSample);
                        maskVal >>= (firstOneBit + 1);
                        indexOffset += (firstOneBit + 1);
                        cnt += 1;
                        firstOneBit = ScalarGetSFFValue<1>(maskVal);
                    }
                    while (firstOneBit != -1 && cnt < numSample) {
                        index = indexOffset + firstOneBit;
                        indice.SetValue(indiceOffset + cnt, index);
                        maskVal >>= (firstOneBit + 1);
                        indexOffset += (firstOneBit + 1);
                        cnt += 1;
                        firstOneBit = ScalarGetSFFValue<1>(maskVal);
                    }
                }
            }
            if (cnt < numSample && lastChunkSize) {
                indexOffset = chunkNum * CHUNK_SIZE;
                Sub(dist, x[chunkNum * CHUNK_SIZE * 8], center[i * 8], 3, lastChunkSize, {1, 1, 1, 1, 1, 0});
                Mul(dist, dist, dist, 3, lastChunkSize, {1, 1, 1, 1, 1, 1});
                WholeReduceSum(dist, dist, 3, lastChunkSize, 1, 1, 1);
                CompareScalar(mask1, dist, minRadius2, CMPMODE::GE, 1, lastChunkSize, {1, 1, 1, 1});
                CompareScalar(mask2, dist, maxRadius2, CMPMODE::LT, 1, lastChunkSize, {1, 1, 1, 1});
                And(mask1, mask1, mask2, lastChunkSize);
                uint64_t maskVal = mask1.ReinterpretCast<uint64_t>().GetValue(0);
                int64_t firstOneBit = ScalarGetSFFValue<1>(maskVal);
                if (firstOneBit != -1) {
                    int32_t index = indexOffset + firstOneBit;
                    if (cnt == 0) {
                        Duplicate(indice[indiceOffset], index, numSample);
                        maskVal >>= (firstOneBit + 1);
                        indexOffset += (firstOneBit + 1);
                        cnt += 1;
                        firstOneBit = ScalarGetSFFValue<1>(maskVal);
                    }
                    while (firstOneBit != -1 && cnt < numSample) {
                        index = indexOffset + firstOneBit;
                        indice.SetValue(indiceOffset + cnt, index);
                        maskVal >>= (firstOneBit + 1);
                        indexOffset += (firstOneBit + 1);
                        cnt += 1;
                        firstOneBit = ScalarGetSFFValue<1>(maskVal);
                    }
                }
            }
        }
        Q_x.FreeTensor(x);
        Q_center.FreeTensor(center);
        Q_indice.EnQue<int32_t>(indice);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t offset = (progress * centerNum + GetBlockIdx() * blockCenterNum) * numSample;
        LocalTensor<int32_t> indice = Q_indice.DeQue<int32_t>();
        DataCopyExtParams copyParam = {(uint16_t)blockCenterNum, uint32_t(numSample * sizeof(float)), 0, 0, 0};
        DataCopyPad(Gm_indice[offset], indice, copyParam);
        Q_indice.FreeTensor(indice);
    }

    __aicore__ inline void PrintParam() {
        printf("=========Tiling=========\n");
        printf("batchSize: %ld\n", batchSize);
        printf("xyzNum: %ld\n", xyzNum);
        printf("centerNum: %ld\n", centerNum);
        printf("numSample: %ld\n", numSample);
        printf("minRadius2: %f\n", minRadius2);
        printf("maxRadius2: %f\n", maxRadius2);
        printf("blockCenterNum: %d\n", blockCenterNum);
        printf("chunkNum: %d\n", chunkNum);
        printf("lastChunkSize: %d\n", lastChunkSize);
        printf("numSampleAlign: %d\n", numSampleAlign);
    }


private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> Q_x, Q_center;
    TQue<TPosition::VECOUT, BUFFER_NUM> Q_indice;
    TBuf<TPosition::VECCALC> B_dist, B_mask1, B_mask2;

    GlobalTensor<float> Gm_x, Gm_center;
    GlobalTensor<int32_t> Gm_indice;

    int32_t batchSize;
    int32_t xyzNum;
    int32_t centerNum, blockCenterNum;
    int32_t numSample;
    float minRadius2;
    float maxRadius2;
    int32_t chunkNum, lastChunkSize;
    int32_t numSampleAlign;
};


class KernelStackBallQuery {
public:
    __aicore__ inline KernelStackBallQuery() {}
    __aicore__ inline void Init(
        GM_ADDR xyz,
        GM_ADDR center_xyz,
        GM_ADDR xyz_batch_cnt,
        GM_ADDR center_xyz_batch_cnt,
        GM_ADDR outputIdx,
        BallQueryTilingData& tiling
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tiling);

        // Set global memory.
        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, totalXyzNum * 3);
        centerxyzGm.SetGlobalBuffer((__gm__ float *)center_xyz, totalCenterXyzNum * 3);
        xyzBatchCntGm.SetGlobalBuffer((__gm__ int32_t *)xyz_batch_cnt, batchSize);
        centerxyzBatchCntGm.SetGlobalBuffer((__gm__ int32_t *)center_xyz_batch_cnt, batchSize);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)outputIdx, totalCenterXyzNum * numSample);

        // Set local memory.
        pipe.InitBuffer(inQueueXYZ, BUFFER_NUM, xyzNum * 8 * sizeof(float));
        pipe.InitBuffer(inQueueCenter, BUFFER_NUM, centerNum * 8 * sizeof(float));
        pipe.InitBuffer(outQueueIndex, BUFFER_NUM, centerNum * numSample * sizeof(int32_t));
        pipe.InitBuffer(tempCenterBuf, 8 * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopRound = batchSize;
        int32_t xyzOffset = 0;
        int32_t centerOffset = 0;
        int32_t indexOffset = 0;
        for (int32_t i = 0; i < loopRound; i++) {
            int32_t curXyzNum = xyzBatchCntGm.GetValue(i);
            int32_t curCenterNum = centerxyzBatchCntGm.GetValue(i);
            // printf("xzyOffset: %d, centerOffset: %d, indexOffset: %d, curXyzNum: %d, curCenterNum: %d"
            //     ,xyzOffset, centerOffset, indexOffset, curXyzNum, curCenterNum);
            CopyIn(xyzOffset, centerOffset, curXyzNum, curCenterNum);
            Compute(i);
            CopyOut(indexOffset, curCenterNum * numSample);
            xyzOffset += (curXyzNum * 3);
            centerOffset += (curCenterNum * 3);
            indexOffset += (curCenterNum * numSample);
        }
    }

private:
    __aicore__ inline void InitTiling(BallQueryTilingData& tiling) {
        this->batchSize = tiling.batchSize;
        this->xyzNum = tiling.xyzNum;
        this->centerNum = tiling.centerNum;
        this->numSample = tiling.numSample;
        this->minRadius2 = tiling.minRadius2;
        this->maxRadius2 = tiling.maxRadius2;

        this->totalXyzNum = tiling.totalXyzNum;
        this->totalCenterXyzNum = tiling.totalCenterXyzNum;
    }
    __aicore__ inline void CopyIn(int32_t xyzOffset, int32_t centerOffset, int32_t xyzLength, int32_t centerLength) {
        LocalTensor<float> xyzLocal = inQueueXYZ.AllocTensor<float>();
        LocalTensor<float> centerLocal = inQueueCenter.AllocTensor<float>();
        DataCopyExtParams copyParams1 {(uint16_t)xyzLength, 3 * sizeof(float), 0, 0, 0};
        DataCopyExtParams copyParams2 {(uint16_t)centerLength, 3 * sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> padParams {true, 0, 5, 0}; 
        DataCopyPad(xyzLocal, xyzGm[xyzOffset], copyParams1, padParams);
        DataCopyPad(centerLocal, centerxyzGm[centerOffset], copyParams2, padParams);
        inQueueXYZ.EnQue<float>(xyzLocal);
        inQueueCenter.EnQue<float>(centerLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> xyzLocal = inQueueXYZ.DeQue<float>();
        LocalTensor<float> centerLocal = inQueueCenter.DeQue<float>();
        LocalTensor<float> tempTensor = tempCenterBuf.Get<float>();
        LocalTensor<int32_t> indexLocal = outQueueIndex.AllocTensor<int32_t>();

        int32_t curCenterNum = centerxyzBatchCntGm.GetValue(progress);
        int32_t curXyzNum = xyzBatchCntGm.GetValue(progress);
        for (int32_t i = 0; i < curCenterNum; i++) {
            LocalTensor<float> center = centerLocal[8 * i];
            LocalTensor<int32_t> index = indexLocal[i * numSample];
            int cnt = 0;
            for (int j = 0; j < curXyzNum; j++) {
                LocalTensor<float> xyz = xyzLocal[8 * j];
                Sub(tempTensor, center, xyz, 8);
                Mul(tempTensor, tempTensor, tempTensor, 8);
                float dist2 = tempTensor.GetValue(0) + tempTensor.GetValue(1) + tempTensor.GetValue(2);
                // printf("dist2: %f\n", dist2);
                // if (progress == 0) {
                //     printf("dist2: %f\n", dist2);
                // }
                if (dist2 < maxRadius2) {
                    if (cnt == 0) {
                        for (int k = 0; k < numSample; k++)
                            index.SetValue(k, j);
                    }
                    index.SetValue(cnt, j);
                    cnt += 1;
                    if (cnt >= numSample) {
                        break;
                    }
                }
            }
        }

        outQueueIndex.EnQue<int32_t>(indexLocal);
        inQueueXYZ.FreeTensor(xyzLocal);
        inQueueCenter.FreeTensor(centerLocal);
    }
    __aicore__ inline void CopyOut(int32_t offset, int32_t length) {
        LocalTensor<int32_t> indexLocal = outQueueIndex.DeQue<int32_t>();
        // for (int i = 0; i < length; i++) {
        //     printf("[copy out] %d\n", indexLocal.GetValue(i));
        // }
        DataCopy(idxGm[offset], indexLocal, length);
        outQueueIndex.FreeTensor(indexLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueXYZ;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCenter;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndex;
    TBuf<> tempCenterBuf;

    GlobalTensor<float> xyzGm;
    GlobalTensor<float> centerxyzGm;
    GlobalTensor<int32_t> xyzBatchCntGm;
    GlobalTensor<int32_t> centerxyzBatchCntGm;
    GlobalTensor<int32_t> idxGm;

    // Tiling data
    int64_t batchSize;
    int64_t xyzNum;
    int64_t centerNum;
    int64_t numSample;
    float minRadius2;
    float maxRadius2;

    int64_t totalXyzNum;
    int64_t totalCenterXyzNum;
};


extern "C" __global__ __aicore__ void ball_query(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch_cnt, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.isStack) {
        KernelStackBallQuery op;
        op.Init(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, idx, tiling_data);
        op.Process();
    }
    else {
        KernelBallQuery<DTYPE_XYZ> op;
        op.Init(xyz, center_xyz, idx, tiling_data);
        op.Process();
    }
    // else if (sizeof(DTYPE_XYZ)) == 2) {
    //     KernelBallQuery<DTYPE_XYZ> op;
    //     op.Init(xyz, center_xyz, idx, tiling_data);
    //     op.Process();
    // }
    // else {
        
    // }
}