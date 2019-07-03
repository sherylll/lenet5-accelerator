
#include "conv.h"
#include <stdio.h>

__global__ void conv_2d_2(
    float data[IN_HEIGHT_3 * IN_WIDTH_3 * N_CHAN_3],
    float res[OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3],
    float weights[FILT_HEIGHT * FILT_WIDTH * N_CHAN_3 * N_FILT_3],
    float biases[N_FILT_3])
{   
    int oh= blockIdx.y * blockDim.y + threadIdx.y;
    int ow= blockIdx.x * blockDim.x + threadIdx.x;
    if (oh>=IN_HEIGHT_3-FILT_HEIGHT+1 || ow>=IN_WIDTH_3-FILT_WIDTH+1)
        return;
    int offset = (oh * OUT_WIDTH_3 + ow)*N_FILT_3;
    for (int ff = 0; ff < N_FILT_3; ff++)
    {
        float temp = biases[ff];
        for (int cc = 0; cc < N_CHAN_3; cc++)
        {
            for (int fh = 0; fh < FILT_HEIGHT; fh++)
            {
                for (int fw = 0; fw < FILT_WIDTH; fw++)
                {
                    int index_weight = fh * FILT_WIDTH * N_CHAN_3 * N_FILT_3 + fw * N_CHAN_3 * N_FILT_3 + cc * N_FILT_3 + ff;
                    // assuming there is no padding
                    //if ((oh + fh) < IN_HEIGHT_3 && (ow + fw) < IN_WIDTH_3)
                        temp += data[((oh + fh) * IN_WIDTH_3 + (ow + fw)) * N_CHAN_3 + cc] * weights[index_weight];
        
                } //end mult loop
            }     //end channel loop
        } //end filter width loop
        res[offset + ff] = (temp > 0)?temp:0;
    }     //end filter height loop
} //end conv2d
