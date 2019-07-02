
// modified by Yuxi Sun
// Keras trained accuracy 98.89%

// comment out to use gpu
#define CPU 

#include "parameters.h"
#include "lenet5.h"
#include "conv.h"

//hls-fpga-machine-learning insert weights
#include "../firmware/weights/w1.h"
#include "../firmware/weights/b1.h"
#include "../firmware/weights/w3.h"
#include "../firmware/weights/b3.h"
#include "../firmware/weights/w5.h"
#include "../firmware/weights/b5.h"
#include "../firmware/weights/w6.h"
#include "../firmware/weights/b6.h"
#include "../firmware/weights/w7.h"
#include "../firmware/weights/b7.h"


void lenet5(
		  input_t data[IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1],
		  result_t res[N_OUTPUTS])
{
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

#ifndef CPU
    float *data_copy, *w1_copy, *b1_copy;
    cudaMallocManaged(&data_copy, IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1 * sizeof(float));
    cudaMallocManaged(&w1_copy, 150 * sizeof(float));
    cudaMallocManaged(&b1_copy, 6 * sizeof(float));
    cudaMemcpy(data_copy, data, sizeof(float)*IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1, cudaMemcpyHostToDevice);
    cudaMemcpy(w1_copy, w1, sizeof(float)*150, cudaMemcpyHostToDevice);
    cudaMemcpy(w1_copy, b1, sizeof(float)*6, cudaMemcpyHostToDevice);

    float *conv2d_layer1_out;
    cudaMallocManaged(&conv2d_layer1_out, OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1 * sizeof(float));
    cudaDeviceSynchronize();

    conv_2d_1<<<1,1>>>(data_copy, conv2d_layer1_out, w1_copy, b1_copy);
    cudaDeviceSynchronize();
#else
    float conv2d_layer1_out[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    conv_2d_1_cpu(data, conv2d_layer1_out, w1, b1);
#endif

    float pool2d_layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    nnet::pooling2d<config2>(conv2d_layer1_out, pool2d_layer2_out);

    float conv2d_layer3_out[OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3];

    conv_2d_2(pool2d_layer2_out, conv2d_layer3_out, w3, b3);

    float layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    nnet::pooling2d<config4>(conv2d_layer3_out, layer4_out);

    float layer5_out[N_LAYER_5];
    nnet::compute_layer<config5>(layer4_out, layer5_out, w5, b5);

    float layer6_out[N_LAYER_6];
    nnet::compute_layer<config6>(layer5_out, layer6_out, w6, b6);

    // float logits7[N_OUTPUTS];

    nnet::compute_layer<config7>(layer6_out, res, w7, b7);

    // todo change to the non-table version of softmax
    // nnet::softmax<float, result_t, softmax_config7>(logits7, res); 

    // cudaFree(data_copy);
    // cudaFree(w1_copy);
    // cudaFree(b1_copy);
    // cudaFree(conv2d_layer1_out);
}
