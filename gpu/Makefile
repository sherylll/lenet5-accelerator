cpu: lenet5_host.cpp lenet5.cu conv.cu lenet5.h conv.h
	nvcc -o lenet5_cpu lenet5_host.cpp lenet5.cu conv.cu -DUSE_CPU
gpu: lenet5_host.cpp lenet5.cu conv.cu lenet5.h conv.h
	nvcc -o lenet5_gpu lenet5_host.cpp lenet5.cu conv.cu
