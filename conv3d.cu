/*
+ * conv_2d_cpu.cpp
 *
 *  Created on: Mar 5, 2018
 *      Author: lyan
 */
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#include <ctime>
#include <iostream>
#include <cstdlib>

#include "common.h"
#include "gpu_conv_3d.h"

using namespace std;

/**
 * convolves in and kernel from top left corner defined by i j k tuple
 */
int convolve(tensor3 in, tensor3 kernel, int i,int j,int k, int kCols, int kRows, int kDepth){

	int result = 0;

	for(int m = 0; m<kCols; m++){
		for(int n = 0; n<kRows; n++){
			for(int p = 0; p<kDepth; p++){
				result += kernel[m][n][p] * in[i + m][j + n][k + p];
			}
		}
	}
	return result;
}


tensor3 conv_3d(tensor3 in, tensor3 kernel, int cols,int rows, int depth, int kCols,int kRows,int kd, int stride, int padding){

	if ((cols + 2 * padding - kCols) % stride != 0){
		cout << "bad stride" << endl;
	}

	if ((rows + 2 * padding - kRows) % stride != 0){
		cout << "bad stride" << endl;
	}

	if ((depth + 2 * padding - kd) % stride != 0){
		cout << "bad stride" << endl;
	}

	int rCols = (cols + 2 * padding - kCols) / stride + 1;
	int rRows = (rows + 2 * padding - kRows) / stride + 1;
	int rDepth = (depth + 2 * padding - kd) / stride + 1;

//	cout << (cols + 2 * padding - kCols) / stride + 1 << endl;
//	cout << (rows + 2 * padding - kRows) / stride + 1 << endl;
//	cout << (depth + 2 * padding - kd) / stride + 1  << endl;

	tensor3 out = initVals(
			(cols + 2 * padding - kCols) / stride + 1,
			(rows + 2 * padding - kRows) / stride + 1,
			(depth + 2 * padding - kd) / stride + 1,
			0);

	tensor3 inPad = pad(in, cols, rows, depth, padding);

	int m = 0, n = 0, l = 0;

	for(int i = 0; m<rCols; i+= stride, m++){
		for(int j = 0; n<rRows; j+= stride, n++){
			for(int k = 0; l<rDepth; k+=stride, l++){
				int t = convolve(inPad, kernel, i,j,k, kCols, kRows, kd);
				out[m][n][l] = t;
			}
			l=0;
		}
		n=0;
	}

	deleteTensor3(inPad, cols + 2 * padding, rows + 2 * padding, depth + 2 * padding);

	return out;
}

void cpu_test(tensor3 large, tensor3 kernel, int dim, int kdim, int padding, int stride, int resDim){

	cout << "starting cpu test" << endl;

	cout << "starting convolution" << endl;
	clock_t start;
	double total = 0;
	int ITER = 2;
	for(int i = 0; i<ITER; i++){
		start = clock();
		tensor3 con = conv_3d(large, kernel, dim,dim,dim,kdim,kdim,kdim,stride,padding);
		total += double(clock() - start) / CLOCKS_PER_SEC;
		deleteTensor3(con, resDim,resDim,resDim);
	}

	cout << "avg time:" << total / ITER << endl;

	cout << "convolution done" << endl;

	tensor3 con = conv_3d(large, kernel, dim,dim,dim,kdim,kdim,kdim,stride,padding);
	printSlice(con, 0, resDim,resDim);
	deleteTensor3(con, resDim,resDim,resDim);

	deleteTensor3(large, dim, dim, dim);
	deleteTensor3(kernel, kdim, kdim, kdim);
}



__host__ void gpu_test(tensor3 large, tensor3 kernel,int dim, int kdim, int padding, int stride, int resDim){

	cout << "starting gpu test" << endl;

	int* g_large = initGpuTensor(large, dim, dim,dim);
	int* g_kernel = initGpuTensor(kernel, kdim,kdim,kdim);

	cout << "gpu ini donet" << endl;

	conv_3d_gpu(large, kernel, dim,dim,dim,kdim,kdim,kdim, padding, stride);

	cout << "conv done" << endl;

	int* out = new int[kdim * kdim * kdim];
	CUDA_CHECK_RETURN(cudaMemcpy(out, g_kernel, t3_int_size(kdim,kdim,kdim), cudaMemcpyDeviceToHost));
	cout << out[0] << " " << out[1] << endl;

	CUDA_CHECK_RETURN(cudaFree(g_large));
	CUDA_CHECK_RETURN(cudaFree(g_kernel));

	deleteTensor3(large, dim, dim, dim);
	deleteTensor3(kernel, kdim, kdim, kdim);

	cout << "gpu test end" << endl;
}

__host__ int main(){
	int dim = 4;
	int kdim = 3;
	int padding = 1;
	int stride = 1;
	int resDim = (dim + 2 * padding - kdim) / stride + 1;

	tensor3 large = initSeq(dim);
	tensor3 kernel = initVals(kdim,1);

	cpu_test(large, kernel, dim, kdim, padding, stride,resDim);
	gpu_test(large, kernel, dim,kdim, padding,stride, resDim);
}
