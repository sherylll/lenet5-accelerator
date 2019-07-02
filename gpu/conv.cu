
#include "conv.h"

__global__ void conv_2d_1(
    float data[IN_HEIGHT_1 * IN_WIDTH_1 * N_CHAN_1],
    float res[OUT_HEIGHT_1 * OUT_WIDTH_1 * N_FILT_1],
    float weights[FILT_HEIGHT * FILT_WIDTH * N_CHAN_1 * N_FILT_1],
    float biases[N_FILT_1])
{
    for (int oh = 0; oh < OUT_HEIGHT_1; oh++)
    {
        for (int ow = 0; ow < OUT_WIDTH_1; ow++)
        {
            for (int ff = 0; ff < N_FILT_1; ff++)
            {
                float temp = 0;
                for (int cc = 0; cc < N_CHAN_1; cc++)
                {
                    for (int fh = 0; fh < FILT_HEIGHT; fh++)
                    {
                        for (int fw = 0; fw < FILT_WIDTH; fw++)
                        {
                            int index_weight = fh * FILT_WIDTH * N_CHAN_1 * N_FILT_1 + fw * N_CHAN_1 * N_FILT_1 + cc * N_FILT_1 + ff;
                            // assuming there is no padding
                            if ((oh + fh) < IN_HEIGHT_1 && (ow + fw) < IN_WIDTH_1)
                                temp += data[((oh + fh) * IN_WIDTH_1 + (ow + fw)) * N_CHAN_1 + cc] * weights[index_weight];
                
                        } //end mult loop
                    }     //end channel loop

                } //end filter width loop
                float res_ = temp + biases[ff];
                res[(oh * OUT_WIDTH_1 + ow) * N_FILT_1 + ff] = (res_ > 0)?res_:0;

            }     //end filter height loop
        }         //end output width loop
    }             //end output height loop
} //end conv2d

void conv_2d_2(
    float data[IN_HEIGHT_3 * IN_WIDTH_3 * N_CHAN_3],
    float res[OUT_HEIGHT_3 * OUT_WIDTH_3 * N_FILT_3],
    float weights[FILT_HEIGHT * FILT_WIDTH * N_CHAN_3 * N_FILT_3],
    float biases[N_FILT_3])
{
    for (int oh = 0; oh < OUT_HEIGHT_3; oh++)
    {
        for (int ow = 0; ow < OUT_WIDTH_3; ow++)
        {
            for (int ff = 0; ff < N_FILT_3; ff++)
            {
                float temp = 0;
                for (int cc = 0; cc < N_CHAN_3; cc++)
                {
                    for (int fh = 0; fh < FILT_HEIGHT; fh++)
                    {
                        for (int fw = 0; fw < FILT_WIDTH; fw++)
                        {
                            int index_weight = fh * FILT_WIDTH * N_CHAN_3 * N_FILT_3 + fw * N_CHAN_3 * N_FILT_3 + cc * N_FILT_3 + ff;
                            // assuming there is no padding
                            if ((oh + fh) < IN_HEIGHT_3 && (ow + fw) < IN_WIDTH_3)
                                temp += data[((oh + fh) * IN_WIDTH_3 + (ow + fw)) * N_CHAN_3 + cc] * weights[index_weight];
                
                        } //end mult loop
                    }     //end channel loop

                } //end filter width loop
                float res_ = temp + biases[ff];
                res[(oh * OUT_WIDTH_3 + ow) * N_FILT_3 + ff] = (res_ > 0)?res_:0;

            }     //end filter height loop
        }         //end output width loop
    }             //end output height loop
} //end conv2d


__host__ void conv_2d_1_cpu(
    float data[IN_HEIGHT_1 * IN_WIDTH_1 * N_CHAN_1],
    float res[OUT_HEIGHT_1 * OUT_WIDTH_1 * N_FILT_1],
    float weights[FILT_HEIGHT * FILT_WIDTH * N_CHAN_1 * N_FILT_1],
    float biases[N_FILT_1])
{
    for (int oh = 0; oh < OUT_HEIGHT_1; oh++)
    {
        for (int ow = 0; ow < OUT_WIDTH_1; ow++)
        {
            for (int ff = 0; ff < N_FILT_1; ff++)
            {
                float temp = 0;
                for (int cc = 0; cc < N_CHAN_1; cc++)
                {
                    for (int fh = 0; fh < FILT_HEIGHT; fh++)
                    {
                        for (int fw = 0; fw < FILT_WIDTH; fw++)
                        {
                            int index_weight = fh * FILT_WIDTH * N_CHAN_1 * N_FILT_1 + fw * N_CHAN_1 * N_FILT_1 + cc * N_FILT_1 + ff;
                            // assuming there is no padding
                            if ((oh + fh) < IN_HEIGHT_1 && (ow + fw) < IN_WIDTH_1)
                                temp += data[((oh + fh) * IN_WIDTH_1 + (ow + fw)) * N_CHAN_1 + cc] * weights[index_weight];
                
                        } //end mult loop
                    }     //end channel loop

                } //end filter width loop
                float res_ = temp + biases[ff];
                res[(oh * OUT_WIDTH_1 + ow) * N_FILT_1 + ff] = (res_ > 0)?res_:0;

            }     //end filter height loop
        }         //end output width loop
    }             //end output height loop
} //end conv2d
