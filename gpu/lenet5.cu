
// modified by Yuxi Sun
// Keras trained accuracy 98.89%

#include "parameters.h"
#include "lenet5.h"
#include "conv.h"
#include "stdio.h"

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

#ifndef USE_CPU
static bool initialized = 0;
static float *d_pool2d_layer2_out;
static float *d_conv2d_layer3_out;
static float *w3_copy, *b3_copy;
static int block_size_1 = 12;
static int num_blocks_1 = (OUT_HEIGHT_1 + block_size_1 - 1)/block_size_1;
static dim3 block(block_size_1,block_size_1);
static dim3 grid (num_blocks_1, num_blocks_1);
#endif

void lenet5(
		  input_t data[IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1],
          result_t res[N_OUTPUTS], bool cleanup)
{

    float conv2d_layer1_out[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    nnet::conv_2d<config1>(data, conv2d_layer1_out, w1, b1);

    float pool2d_layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    nnet::pooling2d<config2>(conv2d_layer1_out, pool2d_layer2_out);
#ifdef USE_CPU
	clock_t begin_time, end_time;
    begin_time = clock();

    float conv2d_layer3_out[OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3];
    nnet::conv_2d<config3>(pool2d_layer2_out, conv2d_layer3_out, w3, b3);

    end_time = clock();
    printf("%f\n", double(end_time - begin_time) / CLOCKS_PER_SEC);
#else
    // prepare memory
    if (!initialized)
    {
        cudaMalloc(&d_pool2d_layer2_out, sizeof(float)*OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2);
        cudaMalloc(&d_conv2d_layer3_out, sizeof(float)*OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3);
        cudaMalloc(&w3_copy, 2400 * sizeof(float));
        cudaMalloc(&b3_copy, 16 * sizeof(float));
        cudaMemcpy(w3_copy, w3, sizeof(float)*2400, cudaMemcpyHostToDevice);
        cudaMemcpy(b3_copy, b3, sizeof(float)*16, cudaMemcpyHostToDevice);    
        initialized = 1;
    }
    
    cudaMemcpy(d_pool2d_layer2_out, pool2d_layer2_out, sizeof(float)*OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2, cudaMemcpyHostToDevice);

    // measure time
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    //Do kernel activity here
    conv_2d_2<<<grid,block>>>(d_pool2d_layer2_out, d_conv2d_layer3_out, w3_copy, b3_copy);

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);

    // device to host
    float conv2d_layer3_out[OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3];
    cudaMemcpy(conv2d_layer3_out, d_conv2d_layer3_out, sizeof(float)*OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3, cudaMemcpyDeviceToHost);
#endif

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
#ifndef USE_CPU
    if (cleanup)
    {
        cudaFree(d_pool2d_layer2_out);
        cudaFree(conv2d_layer1_out);
        cudaFree(w3_copy);
        cudaFree(b3_copy);
    }
#endif
}
