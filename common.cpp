/*
 * common.h
 *
 *  Created on: Mar 9, 2018
 *      Author: lyan
 */

#include "common.h"
#include <cstdlib>

using namespace std;

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void CheckCudaErrorAux(const char *file, unsigned line,
                const char *statement, cudaError_t err) {
        if (err == cudaSuccess)
                return;
        std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
                        << err << ") at " << file << ":" << line << std::endl;
        exit(1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


int t3_int_size(int d1,int d2, int d3){
	return sizeof(int) * d1 * d2 * d3;
}

int* flat(tensor3 t, int cols,int rows, int depth){
	int* res = new int[cols * rows * depth];

	int cnt = 0;

	for(int k = 0; k<depth; k++){
		for(int j = 0; j<rows; j++){
			for(int i = 0; i< cols; i++){
				res[cnt++] = t[k][j][i];
			}
		}
	}

	return res;
}


void printSlice(tensor3 t, int pos, int rows, int cols){
	for(int i = 0; i<cols; i++){
		for(int j = 0; j<rows; j++){
			cout << t[i][j][pos] << " ";
		}
		cout << endl;
	}
}

tensor3 initSeq(int len){
	tensor3 res = new int**[len];

	int seq = 0;

	for(int i = 0; i<len; i++){
		res[i] = new int*[len];
		for(int j = 0; j<len; j++){
			res[i][j] = new int[len];
			for(int k = 0; k<len; k++){
				res[i][j][k] = seq++;
			}
		}
	}

	return res;
}

tensor3 initVals(int len, int val){
	tensor3 res = new int**[len];

	for(int i = 0; i<len; i++){
		res[i] = new int*[len];
		for(int j = 0; j<len; j++){
			res[i][j] = new int[len];
			for(int k = 0; k<len; k++){
				res[i][j][k] = val;
			}
		}
	}

	return res;
}

tensor3 initVals(int cols, int rows, int depth, int val){
	tensor3 res = new int**[cols];

	for(int i = 0; i<cols; i++){
		res[i] = new int*[rows];
		for(int j = 0; j<rows; j++){
			res[i][j] = new int[depth];
			for(int k = 0; k<depth; k++){
				res[i][j][k] = val;
			}
		}
	}

	return res;
}

void deleteTensor3(tensor3 t, int d1,int d2, int d3){
	for(int i = 0; i<d1; i++){
		for(int j = 0; j<d2; j++){
			delete [] t[i][j];
		}
		delete [] t[i];
	}
	delete t;
}

tensor3 pad(tensor3 in, int cols, int rows, int depth, int pad){

	int rCols = cols + 2 * pad;
	int rRows = rows + 2 * pad;
	int rd = depth + 2 * pad;

	tensor3 res = initVals(rCols, rRows, rd, 0);

	for(int i = pad; i<rCols - pad; i++){
		for(int j = pad; j<rRows - pad; j++){
			for(int k = pad; k<rd - pad; k++){
				res[i][j][k] = in[i-pad][j-pad][k-pad];
			}
		}
	}
	return res;
}


bool equal(tensor3 t1, tensor3 t2, int cols, int rows, int depth){
	for(int i = 0; i<rows; i++){
		for(int j = 0; j<cols; j++){
			for(int k = 0; k<depth; k++){
				if (t1[i][j][k] != t2[i][j][k]){
					return false;
				}
			}
		}
	}
	return true;
}
