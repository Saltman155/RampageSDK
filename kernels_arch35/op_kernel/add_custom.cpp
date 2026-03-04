/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void SimtAddCompute(
    __gm__ T* dst, __gm__ T* x, __gm__ T* y, uint32_t totalLength) {
    int begin = Simt::GetThreadIdx<0>() + Simt::GetBlockIdx() * Simt::GetThreadNum<0>();
    int step = Simt::GetThreadNum<0>() * Simt::GetBlockNum();
    for (int i = begin; i < totalLength; i += step) {
        dst[i] = x[i] + y[i];
    }
}

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        this->totalLength = totalLength;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_X *)z, totalLength);
    }
    __aicore__ inline void Process()
    {
        __gm__ DTYPE_X* xPtr = (__gm__ DTYPE_X*)xGm.GetPhyAddr();
        __gm__ DTYPE_X* yPtr = (__gm__ DTYPE_X*)yGm.GetPhyAddr();
        __gm__ DTYPE_X* zPtr = (__gm__ DTYPE_X*)zGm.GetPhyAddr();
        Simt::VF_CALL<SimtAddCompute<DTYPE_X>>(Simt::Dim3{128, 1, 1},
            zPtr, xPtr, yPtr, this->totalLength);
    }

private:
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> yGm;
    GlobalTensor<DTYPE_X> zGm;
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(addCustomTiling, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, addCustomTiling.totalLength);
    op.Process();
}
