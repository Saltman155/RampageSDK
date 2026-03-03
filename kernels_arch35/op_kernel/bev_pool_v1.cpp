/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"


using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void SimtCompute(
    __gm__ T* dst, __gm__ T* feat, __gm__ int32_t* geom, uint32_t n, uint32_t b, uint32_t d, uint32_t h, uint32_t w, uint32_t c) {
    int begin = Simt::GetThreadIdx<0>() + Simt::GetBlockIdx() * Simt::GetThreadNum<0>();
    int step = Simt::GetThreadNum<0>() * Simt::GetBlockNum();
    uint32_t geom_b_offset, geom_d_offset, geom_h_offset, geom_w_offset, geom_offset;
    for (int i = begin; i < n; i += step){
        // geom : [N, 4] , 其中4分别代表 h,w,b,d 的索引
        geom_b_offset = geom[i * 4 + 2] * (c * d * h * w );
        geom_d_offset = geom[i * 4 + 3] * (c * h * w);
        geom_h_offset = geom[i * 4] * (c * w);
        geom_w_offset = geom[i * 4 + 1] * c;
        geom_offset = geom_b_offset + geom_d_offset + geom_h_offset + geom_w_offset;
        for (int j = 0; j < c; j++){
            uint32_t feat_offset = i * c + j;
            // 这里可以用非原子加来测试多线程的散列写性能，实际上如果没有对offset做排序的话，会有冲突，所以必须用原子加
            // 如果非原子加的散列写性能很不错，可以考虑先对geom做排序，再进行非原子加散列写
            // dst[geom_offset + j] += feat[feat_offset];
            AtomicAdd(&dst[geom_offset + j], feat[feat_offset]);
        }
    }
}


class BevPoolKernel {
public:
    __aicore__ inline BevPoolKernel() {}
    __aicore__ inline void Init(GM_ADDR featDevice, GM_ADDR geomFeatDevice, GM_ADDR outputDevice, 
        uint32_t totalLength, uint32_t N, uint32_t B, uint32_t D, uint32_t H, uint32_t W, uint32_t C)
    {
        this->totalLength = totalLength;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->N = N;
        this->B = B;
        this->D = D;
        this->H = H;
        this->W = W;
        this->C = C;
        featGm.SetGlobalBuffer((__gm__ float *)featDevice, N * C);
        geomGm.SetGlobalBuffer((__gm__ int32_t *)geomFeatDevice, N * 4);
        outputGm.SetGlobalBuffer((__gm__ float *)outputDevice, B * C * D * H * W);
    }

    __aicore__ inline void Process()
    {
        __gm__ float* featPtr = (__gm__ float*) featGm.GetPhyAddr();
        __gm__ int32_t* geomPtr = (__gm__ int32_t*) geomGm.GetPhyAddr();
        __gm__ float* outputPtr = (__gm__ float*) outputGm.GetPhyAddr();
        Simt::VF_CALL<SimtCompute<float>>(Simt::Dim3{128, 1, 1}, 
            outputPtr, featPtr, geomPtr, this->N, this->B, this->D, this->H, this->W, this->C);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
    }
    __aicore__ inline void Compute(int32_t progress)
    {
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> featGm;
    AscendC::GlobalTensor<int32_t> geomGm;
    AscendC::GlobalTensor<float> outputGm;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t N, B, D, H, W, C; 

};

extern "C" __global__ __aicore__ void bev_pool_v1(GM_ADDR featDevice, GM_ADDR geomFeatDevice, GM_ADDR outputDevice, GM_ADDR workspace, GM_ADDR tiling)
{

    GET_TILING_DATA(bevPoolTiling, tiling);
    BevPoolKernel op;

    op.Init(
        featDevice, 
        geomFeatDevice, 
        outputDevice, 
        bevPoolTiling.totalLength,
        bevPoolTiling.N,
        bevPoolTiling.B,
        bevPoolTiling.D,
        bevPoolTiling.H,
        bevPoolTiling.W,
        bevPoolTiling.C
    );
    op.Process();
}
