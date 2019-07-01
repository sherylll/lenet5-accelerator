
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cstring>
#include <ctime>
#include "parameters.h"
#include "lenet5.h"

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
	clock_t begin_time, end_time;
	double total_time = 0;

	for (int im=0; im < TEST_SIZE; im ++){
			sprintf(x_str, "%d.txt", im);
			std::string image_path = "../test_images/";
			image_path += std::string(x_str);
			strcpy(path_cstr, image_path.c_str());
			if (read_to_array(path_cstr, data_str, &y_test) == 0){
				unsigned short size_in, size_out;
				begin_time = clock();
				lenet5(data_str, probs);
				end_time = clock();
				total_time += double(end_time - begin_time) / CLOCKS_PER_SEC;
				int y_pred = max_likelihood(probs);
				// std::cout << im << " " << (y_pred == y_test)<< std::endl;
				if (y_pred == y_test)
					counter++;
			}
			else
				std::cout << "failed to read file" << std::endl;
	}

	std::cout << "(partial) accuracy: " <<  counter/(float)TEST_SIZE << std::endl;
	std::cout << "average latency (inference/s): " << total_time/TEST_SIZE << std::endl;
	return 0;
}
