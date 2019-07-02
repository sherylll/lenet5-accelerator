//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_H_
#define NNET_H_

#include <cstdlib>

namespace nnet
{

struct conv2d_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;
};

template <typename CONFIG_T>
void conv_2d(
    float data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    float res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    float weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    float biases[CONFIG_T::n_filt])
{
    for (int oh = 0; oh < CONFIG_T::out_height; oh++)
    {
        for (int ow = 0; ow < CONFIG_T::out_width; ow++)
        {
            for (int ff = 0; ff < CONFIG_T::n_filt; ff++)
            {
                float temp = 0;
                for (int cc = 0; cc < CONFIG_T::n_chan; cc++)
                {
                    for (int fh = 0; fh < CONFIG_T::filt_height; fh++)
                    {
                        for (int fw = 0; fw < CONFIG_T::filt_width; fw++)
                        {
                            int index_weight = fh * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt + fw * CONFIG_T::n_chan * CONFIG_T::n_filt + cc * CONFIG_T::n_filt + ff;
                            // assuming there is no padding
                            if ((oh + fh) < CONFIG_T::in_height && (ow + fw) < CONFIG_T::in_width)
                                temp += data[((oh + fh) * CONFIG_T::in_width + (ow + fw)) * CONFIG_T::n_chan + cc] * weights[index_weight];
                
                        } //end mult loop
                    }     //end channel loop

                } //end filter width loop
                float res_ = temp + biases[ff];
                res[(oh * CONFIG_T::out_width + ow) * CONFIG_T::n_filt + ff] = (res_ > 0)?res_:0;

            }     //end filter height loop
        }         //end output width loop
    }             //end output height loop
} //end conv2d

//////////// pool2d ////////////////
struct pooling2d_config
{
    // IO size
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_filt = 4;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned out_height = (in_height - pool_height) / stride_height + 1;
    static const unsigned out_width = (in_width - pool_width) / stride_width + 1;
    // Padding
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
};

template <typename CONFIG_T>
void pooling2d(float data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
               float res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt])
{

    // Add any necessary padding
    const unsigned padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
    const unsigned padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++)
    {
        float pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
        // Loop over input image y in steps of stride
        for (int ii = 0; ii < padded_height; ii += CONFIG_T::stride_height)
        {
            // Loop over input image x in steps of stride
            for (int jj = 0; jj < padded_width; jj += CONFIG_T::stride_width)
            {
                // Keep track of number of pixels in image vs padding region
                unsigned img_overlap = 0;
                // Loop over pool window y
                for (int kk = 0; kk < CONFIG_T::stride_height; kk++)
                {
                    // Loop over pool window x
                    for (int ll = 0; ll < CONFIG_T::stride_width; ll++)
                    {
                        if (ii + kk < CONFIG_T::pad_top || ii + kk >= (padded_height - CONFIG_T::pad_bottom) || jj + ll < CONFIG_T::pad_left || jj + ll >= (padded_width - CONFIG_T::pad_right))
                        {
                            // Add padding
                            pool[kk * CONFIG_T::stride_width + ll] = 0;
                        }
                        else
                        {
                            pool[kk * CONFIG_T::stride_width + ll] = data[((ii + kk)*CONFIG_T::in_width + (jj + ll))*CONFIG_T::n_filt + ff];
                            img_overlap++;
                        }
                    }
                }
                // do the pooling
                float max_pool = pool[0];
                for (int i = 1; i < N; i++)
                {
                    max_pool = pool[i] > max_pool ? pool[i] : max_pool;
                }
                res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt + (jj / CONFIG_T::stride_width) * CONFIG_T::n_filt + ff] = max_pool;

            }
        }
    }
}

////////////// fully connected ///////////////////
struct layer_config
{
    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;
};

template <typename CONFIG_T>
void compute_layer(
    float data[CONFIG_T::n_in],
    float res[CONFIG_T::n_out],
    float weights[CONFIG_T::n_in * CONFIG_T::n_out],
    float biases[CONFIG_T::n_out])
{
    float cache;
    float acc[CONFIG_T::n_out];

    // Initialize accumulator with input biases
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++)
    {
        acc[iacc] = (float)biases[iacc];
    }

    // Do the matrix-multiply
    for (int ii = 0; ii < CONFIG_T::n_in; ii++)
    {
        cache = data[ii];
        for (int jj = 0; jj < CONFIG_T::n_out; jj++)
        {
            int index = ii * CONFIG_T::n_out + jj;
            float mult = cache * weights[index];
            acc[jj] += mult;
        }
    }
    // Cast to "float" type
    for (int ires = 0; ires < CONFIG_T::n_out; ires++)
    {
        if (acc[ires] > 0)
            res[ires] = (float)(acc[ires]);
        else
            res[ires] = 0;
    }
}

} // namespace nnet

#endif
