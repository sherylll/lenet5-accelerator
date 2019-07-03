
#ifndef LENET5_H_
#define LENET5_H_

#include "parameters.h"
#include "nnet.h"

void lenet5(
      input_t data[IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1],
      result_t res[N_OUTPUTS], bool cleanup = false);


#endif

