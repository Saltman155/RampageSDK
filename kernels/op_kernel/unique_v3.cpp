#include "kernel_operator.h"
#include "unique_v3.h"
#include "unique_v3_counts.h"
#include "unique_v3_inverse.h"


extern "C" __global__ __aicore__ void unique_v3(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt, 
    GM_ADDR inverse, GM_ADDR counts, 
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelUnique<DTYPE_INPUT> op(pipe);
    op.Init(input,
            output,
            uniqueCnt,
            inverse,
            counts,
            workspace,
            tiling_data.totalLength,
            tiling_data.shortBlockTileNum,
            tiling_data.tileLength,
            tiling_data.tailLength,
            tiling_data.aivNum,
            tiling_data.blockNum,
            tiling_data.shortBlockNum,
            tiling_data.flagInverse,
            tiling_data.flagCounts);
    op.Process();
}
