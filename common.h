#define COMMON_H_

typedef int*** tensor3;
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#include <iostream>

void CheckCudaErrorAux(const char *file, unsigned line,
                const char *statement, cudaError_t err);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

int t3_int_size(int d1,int d2, int d3);
int* flat(tensor3 t, int d1,int d2, int d3);
void printSlice(tensor3 t, int pos, int rows, int cols);
tensor3 initSeq(int len);
tensor3 initVals(int len, int val);
tensor3 initVals(int cols, int rows, int depth, int val);
void deleteTensor3(tensor3 t, int d1,int d2, int d3);
tensor3 pad(tensor3 in, int cols, int rows, int depth, int pad);
