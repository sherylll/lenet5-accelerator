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
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cstring>
#include "firmware/parameters.h"
#include "firmware/lenet5.h"
#include "nnet_helpers.h"

#define IMAGE_WIDTH 28
#define TEST_SIZE 100 // full test set has 10000 samples



int max_likelihood(result_t y[N_OUTPUTS])
{
	int i_likely = 0;
	result_t y_max = 0;
	for (int i = 0; i < N_OUTPUTS; i++)
	{
		if (y[i] > y_max)
		{
			y_max = y[i];
			i_likely = i;
		}
	}
	return i_likely;
}

int read_to_array(char *path, input_t x_test[IMAGE_WIDTH*IMAGE_WIDTH*1], int *y_test)
{
	std::ifstream inFile;
	inFile.open(path);
	if (!inFile)
		return -1;
	if (inFile.get() == '#')
		inFile >> *y_test;
//	std::cout << *y_test;
	for (int i = 0; i < IMAGE_WIDTH; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			inFile >> x_test[i*IMAGE_WIDTH+j+0];
		}
	}
	inFile.close();
	return 0;
}

int main(int argc, char **argv)
{

	input_t  data_str[IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1];

	result_t probs[N_OUTPUTS] = {0};
	int y_test, counter = 0;

	char x_str[10] = "";
	char path_cstr[30];

	for (int im=0; im < TEST_SIZE; im ++){
			sprintf(x_str, "%d.txt", im);
			std::string image_path = "test_images/";
			image_path += std::string(x_str);
			strcpy(path_cstr, image_path.c_str());
			if (read_to_array(path_cstr, data_str, &y_test) == 0){
				unsigned short size_in, size_out;
				lenet5(data_str, probs);

				int y_pred = max_likelihood(probs);
				std::cout << im << " " << (y_pred == y_test)<< std::endl;
				if (y_pred == y_test)
					counter++;
			}
			else
				std::cout << "failed to read file" << std::endl;
	}
	std::cout << counter;


	return 0;
}
