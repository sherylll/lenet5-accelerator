#include "parameters.h"
#include "nnet.h"

__global__ void conv_2d_1(
    float data[IN_HEIGHT_1 * IN_WIDTH_1 * N_CHAN_1],
    float res[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1],
    float weights[FILT_HEIGHT * FILT_WIDTH * N_CHAN_1 * N_FILT_1],
    float biases[N_FILT_1]);

void conv_2d_2(
    float data[IN_HEIGHT_3 * IN_WIDTH_3 * N_CHAN_3],
    float res[OUT_HEIGHT_3 * OUT_WIDTH_3*N_FILT_3],
    float weights[FILT_HEIGHT * FILT_WIDTH * N_CHAN_3 * N_FILT_3],
    float biases[N_FILT_3]);    