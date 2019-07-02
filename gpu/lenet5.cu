
// modified by Yuxi Sun
// Keras trained accuracy 98.89%

// comment out to use gpu
#define USE_CPU 

#include "parameters.h"
#include "lenet5.h"
// #include "conv.h"

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

__host__ void kernel_cpu(input_t data[IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1],
    result_t res[N_OUTPUTS])
{
    float conv2d_layer1_out[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    nnet::conv_2d<config1>(data, conv2d_layer1_out, w1, b1);

    float pool2d_layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    nnet::pooling2d<config2>(conv2d_layer1_out, pool2d_layer2_out);

    float conv2d_layer3_out[OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3];
    nnet::conv_2d<config3>(pool2d_layer2_out, conv2d_layer3_out, w3, b3);

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
}

__global__ void kernel(input_t data[IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1],
    result_t res[N_OUTPUTS])
{
    float conv2d_layer1_out[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    nnet::conv_2d<config1>(data, conv2d_layer1_out, w1, b1);

    float pool2d_layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    nnet::pooling2d<config2>(conv2d_layer1_out, pool2d_layer2_out);

    float conv2d_layer3_out[OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3];
    nnet::conv_2d<config3>(pool2d_layer2_out, conv2d_layer3_out, w3, b3);

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
}

void lenet5(input_t data[IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1],
		  result_t res[N_OUTPUTS])
{
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

#ifndef USE_CPU
    // data
    float *data_copy; 
    cudaMallocManaged(&data_copy, IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1 * sizeof(float));
    cudaMemcpy(data_copy, data, sizeof(float)*IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1, cudaMemcpyHostToDevice);

    // layer1
    float *w1_copy, *b1_copy;
    cudaMallocManaged(&w1_copy, 150 * sizeof(float));
    cudaMallocManaged(&b1_copy, 6 * sizeof(float));
    cudaMemcpy(w1_copy, w1, sizeof(float)*150, cudaMemcpyHostToDevice);
    cudaMemcpy(b1_copy, b1, sizeof(float)*6, cudaMemcpyHostToDevice);

    // layer 3
    float *w3_copy, *b3_copy;
    cudaMallocManaged(&w3_copy, 2400 * sizeof(float));
    cudaMallocManaged(&b3_copy, 16 * sizeof(float));
    cudaMemcpy(w3_copy, w3, sizeof(float)*2400, cudaMemcpyHostToDevice);
    cudaMemcpy(b3_copy, b3, sizeof(float)*16, cudaMemcpyHostToDevice);

    // layer 5
    float *w5_copy, *b5_copy;
    cudaMallocManaged(&w5_copy, 30720 * sizeof(float));
    cudaMallocManaged(&b5_copy, 120 * sizeof(float));
    cudaMemcpy(w5_copy, w5, sizeof(float)*30720, cudaMemcpyHostToDevice);
    cudaMemcpy(b5_copy, b5, sizeof(float)*120, cudaMemcpyHostToDevice);

    // layer 6
    float *w6_copy, *b6_copy;
    cudaMallocManaged(&w6_copy, 10080 * sizeof(float));
    cudaMallocManaged(&b6_copy, 84 * sizeof(float));
    cudaMemcpy(w6_copy, w6, sizeof(float)*10080, cudaMemcpyHostToDevice);
    cudaMemcpy(b6_copy, b6, sizeof(float)*84, cudaMemcpyHostToDevice);

    // layer 7
    float *w7_copy, *b7_copy;
    cudaMallocManaged(&w7_copy, 840 * sizeof(float));
    cudaMallocManaged(&b7_copy, 10 * sizeof(float));
    cudaMemcpy(w7_copy, w7, sizeof(float)*840, cudaMemcpyHostToDevice);
    cudaMemcpy(b7_copy, b7, sizeof(float)*10, cudaMemcpyHostToDevice);
    
    // result
    float *res_copy;
    cudaMallocManaged(&res_copy, N_OUTPUTS * sizeof(float));

    // sync
    cudaDeviceSynchronize();

    // int block_size_1 = 32;
    // int num_blocks_1 = (OUT_HEIGHT_1 + block_size_1 - 1)/block_size_1;
    kernel<<<1,1>>>(data_copy, res_copy);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(res, res_copy, sizeof(float)*N_OUTPUTS, cudaMemcpyDeviceToHost);

    // clean up
    cudaFree(data_copy);
    cudaFree(w1_copy);
    cudaFree(b1_copy);
    cudaFree(w3_copy);
    cudaFree(b3_copy);
    cudaFree(w5_copy);
    cudaFree(b5_copy);
    cudaFree(w6_copy);
    cudaFree(b6_copy);
    cudaFree(w7_copy);
    cudaFree(b7_copy);
#else
    kernel_cpu(data, res);
#endif
}
