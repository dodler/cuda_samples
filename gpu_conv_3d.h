/*
 * gpu_conv_3d.h
 *
 *  Created on: Mar 9, 2018
 *      Author: lyan
 */

#ifndef GPU_CONV_3D_H_
#define GPU_CONV_3D_H_

#include "common.h"

int* initGpuTensor(tensor3 t,int d1,int d2, int d3);
__global__ void conv_3d_gpu(int* in, int* kernel, int* out, int cols,int rows,int depth, int stride,
		int kCols, int kRows, int kDepth);
__host__ tensor3 conv_3d_gpu(tensor3 in, tensor3 kernel, int cols, int rows, int depth,
		int kCols, int kRows, int kDepth, int padding, int stride);


#endif /* GPU_CONV_3D_H_ */
