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
/////////////////////////////////////////////////////////////////////////////
// modified by Yuxi Sun
// Keras trained accuracy 98.89%

#include "parameters.h"
#include "lenet5.h"

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

    layer1_t conv2d_layer1_out[OUT_HEIGHT_1][OUT_WIDTH_1][N_FILT_1];

    nnet::conv_2d<input_t, layer1_t, config1>(data, conv2d_layer1_out, w1, b1);

    layer1_t pool2d_layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    nnet::pooling2d<layer1_t, config2>(conv2d_layer1_out, pool2d_layer2_out);

    layer3_t conv2d_layer3_out[OUT_HEIGHT_3][OUT_WIDTH_3][N_FILT_3];

    nnet::conv_2d<layer2_t, layer3_t, config3>(pool2d_layer2_out, conv2d_layer3_out, w3, b3);

    layer3_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    nnet::pooling2d<layer3_t, config4>(conv2d_layer3_out, layer4_out);

    layer5_t layer5_out[N_LAYER_5];
    nnet::compute_layer<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5);

    layer6_t layer6_out[N_LAYER_6];
    nnet::compute_layer<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    layer6_t logits7[N_OUTPUTS];

    nnet::compute_layer<layer6_t, layer6_t, config7>(layer6_out, res, w7, b7);

    // todo change to the non-table version of softmax
    // nnet::softmax<layer6_t, result_t, softmax_config7>(logits7, res); 

}
