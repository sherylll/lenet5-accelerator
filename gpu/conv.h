#include "parameters.h"
#include "nnet.h"

__global__ void conv_2d_2(
    float data[IN_HEIGHT_3 * IN_WIDTH_3 * N_CHAN_3],
    float res[OUT_HEIGHT_3 * OUT_WIDTH_3*N_FILT_3],
    float weights[FILT_HEIGHT * FILT_WIDTH * N_CHAN_3 * N_FILT_3],
    float biases[N_FILT_3]);    
