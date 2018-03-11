
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include <ctime>

#include "common.h"
#include "gpu_conv_3d.h"

static const int WORK_SIZE = 256;

using namespace std;

int* initGpuTensor(tensor3 t,int d1,int d2, int d3){
	int* res;
	int* f = flat(t,d1,d2,d3);
	CUDA_CHECK_RETURN(cudaMalloc((void**)&res, t3_int_size(d1,d2,d3)));
	CUDA_CHECK_RETURN(cudaMemcpy(res, f, t3_int_size(d1,d2,d3), cudaMemcpyHostToDevice));

	delete [] f;

	return res;
}

tensor3 fromGpu(int* g_tensor, int cols, int rows, int depth){
	tensor3 res = initVals(cols, rows, depth, 0);

	int* cf_tensor = new int[cols * rows * depth];

	CUDA_CHECK_RETURN(cudaMemcpy(cf_tensor, g_tensor, t3_int_size(cols, rows, depth), cudaMemcpyDeviceToHost));

	cout << "mem cpy done" << endl;

	for(int k = 0; k<depth; k++){
		for(int j = 0; j<rows; j++){
			for(int i = 0; i<cols; i++){
				res[k][i][j] = cf_tensor[k *  cols * rows + i * cols + j];
			}
		}
	}

	cout << "copy done" << endl;

	delete [] cf_tensor;

	return res;
}

__global__ void conv_3d_gpu(int* in, int* kernel, int* out, int cols,int rows,int depth, int padding, int stride,
		int kCols, int kRows, int kDepth){

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int num_thr = blockDim.x;
	int num_blocks = gridDim.x;
	int block_pos = blockDim.x;

	int p_cols = cols + 2 * padding;
	int p_rows = rows + 2 * padding;

	int cols_per_block = cols / num_thr;
	int rows_per_block = rows / num_blocks;
	int resid = cols - cols_per_block * num_thr;
	int resid_rows = rows - rows_per_block * num_blocks;

	__shared__ int shared_in[4096];

	int cnt = 0;

	for(int k = 0; k<depth; k+=stride)
		for(int j = 0; j<rows; j+= stride)
			for(int i = pos * cols_per_block; i<(pos+1) * cols_per_block; i+= stride)
				for(int l = 0; l<kDepth; l++)
					for(int n = 0; n<kRows; n++)
						for(int m = 0; m<kCols; m++)
							shared_in[cnt++] = in[(k+l) * p_cols * p_rows + (j+n) * p_cols + (i+m)];

	for(int k = 0; k<depth; k+=stride){
		for(int j = block_pos * rows_per_block; j<(block_pos+1) * rows_per_block; j+= stride){
			for(int i = pos * cols_per_block; i<(pos+1) * cols_per_block; i+= stride){
//			for(int i = 0; i<cols; i+= stride){

				int t = 0;
				for(int l = 0; l<kDepth; l++){
					for(int n = 0; n<kRows; n++){
						for(int m = 0; m<kCols; m++){
							t += shared_in[(k+l) * p_cols * p_rows + (j+n) * p_cols + (i+m)] *
									kernel[l * kRows * kCols + n * kCols + m];
						}
					}
				}
				out[k * cols * rows + j * cols + i] = t;
//				out[cnt++] = threadIdx.x;
			}
		}
	}

	if (pos == 0){
		for(int k = 0; k<depth; k+=stride){
			for(int j = 0; j<rows; j+= stride){
				for(int i = cols - resid; i<cols; i+= stride){

					int t = 0;
					for(int l = 0; l<kDepth; l++){
						for(int n = 0; n<kRows; n++){
							for(int m = 0; m<kCols; m++){
								t += in[(k+l) * p_cols * p_rows + (j+n) * p_cols + (i+m)] *
											kernel[l * kRows * kCols + n * kCols + m];
							}
						}
					}
					out[k * cols * rows + j * cols + i] = t;
				}
			}
		}
		for(int k = 0; k<depth; k+=stride){
			for(int j = rows - resid_rows; j<rows; j+= stride){
				for(int i = 0; i<cols; i+= stride){

					int t = 0;
					for(int l = 0; l<kDepth; l++){
						for(int n = 0; n<kRows; n++){
							for(int m = 0; m<kCols; m++){
								t += in[(k+l) * p_cols * p_rows + (j+n) * p_cols + (i+m)] *
									 kernel[l * kRows * kCols + n * kCols + m];
							}
						}
					}
					out[k * cols * rows + j * cols + i] = t;
				}
			}
		}
	}
}

__host__ tensor3 conv_3d_gpu(tensor3 in, tensor3 kernel, int cols, int rows, int depth,
		int kCols, int kRows, int kDepth, int padding, int stride){

		if ((cols + 2 * padding - kCols) % stride != 0){
			cout << "bad stride" << endl;
		}

		if ((rows + 2 * padding - kRows) % stride != 0){
			cout << "bad stride" << endl;
		}

		if ((depth + 2 * padding - kDepth) % stride != 0){
			cout << "bad stride" << endl;
		}

		int rCols = (cols + 2 * padding - kCols) / stride + 1;
		int rRows = (rows + 2 * padding - kRows) / stride + 1;
		int rDepth = (depth + 2 * padding - kDepth) / stride + 1;

		tensor3 out = initVals(rCols,rRows,rDepth,0);

		tensor3 inPad = pad(in, cols, rows, depth, padding);

		cout << "init gpu start" << endl;

		int* g_in = initGpuTensor(inPad, cols + 2 * padding, rows + 2 * padding, depth + 2 * padding);
		int* g_kernel = initGpuTensor(kernel, kCols, kRows, kDepth);
		int* g_out = initGpuTensor(out, rCols, rRows, rDepth);

		cout << "starting convolution" << endl;

		double total = 0;

		for(int i = 0; i<ITER; i++){
			GpuTimer timer;
			timer.Start();

			conv_3d_gpu<<<32,32>>>(g_in, g_kernel, g_out, cols, rows, depth, padding, stride, kCols, kRows, kDepth);

			timer.Stop();
			total += timer.Elapsed();

		}

		cout << "gpu avg time:" << total / ITER << endl;

		cudaDeviceSynchronize();

		tensor3 c_out = fromGpu(g_out, rCols, rRows, rDepth);

		cout << "from gpu done" << endl;

//		printSlice(c_out, 0, rRows, rCols);
//		cout << "-------------------------" << endl;
//		printSlice(c_out, 1, rRows, rCols);
//		cout << "-------------------------" << endl;
//		printSlice(c_out, 2, rRows, rCols);
//		cout << "-------------------------" << endl;
//		printSlice(c_out, 3, rRows, rCols);
//		cout << "-------------------------" << endl;

		CUDA_CHECK_RETURN(cudaFree(g_in));
		CUDA_CHECK_RETURN(cudaFree(g_kernel));
		CUDA_CHECK_RETURN(cudaFree(g_out));

		deleteTensor3(inPad, cols + 2 * padding, rows + 2 * padding, depth + 2 * padding);
		return c_out;
}
