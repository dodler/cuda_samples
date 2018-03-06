/*
+ * conv_2d_cpu.cpp
 *
 *  Created on: Mar 5, 2018
 *      Author: lyan
 */

typedef int*** tensor3;

#include <iostream>

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

	int seq = 0;

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

	int seq = 0;

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

	int seq = 0;

	int rCols = cols + 2 * pad;
	int rRows = rows + 2 * pad;
	int rd = depth + 2 * pad;

	cout << "padding result:" << rCols << " " << rRows << " " << rd << endl;

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

	cout << (cols + 2 * padding - kCols) / stride + 1 << endl;
	cout << (rows + 2 * padding - kRows) / stride + 1 << endl;
	cout << (depth + 2 * padding - kd) / stride + 1  << endl;

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
	return out;
}


int main(int argc, char** argv){
	tensor3 test = initSeq(3);

	tensor3 large = initSeq(27);
	tensor3 kernel = initVals(3,2);

	printSlice(kernel, 0,3,3);

	tensor3 p = pad(kernel, 3,3,3, 1);

	//printSlice(p, 0, 5,5);

	cout << "starting convolution" << endl;
	int c = convolve(test,kernel, 0,0,0,3,3,3);
	tensor3 con = conv_3d(large, kernel, 27,27,27,3,3,3,1,0);

	cout << "convolution done" << endl;

	printSlice(test,2,3,3);
	printSlice(con, 2, 11,11);

	cout << c << endl;

	deleteTensor3(con, 11,11,11);
	deleteTensor3(p, 5,5,5);
	deleteTensor3(test, 27,27,27);
	deleteTensor3(kernel, 3,3,3);
}
