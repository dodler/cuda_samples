/*
 * image_statistic.h
 *
 *  Created on: May 8, 2016
 *      Author: dodler
 */

#ifndef IMAGE_STATISTIC_H_
#define IMAGE_STATISTIC_H_

struct matrix {
	int cols, rows;
	int* values;

	matrix(int cols, int rows) {
		this->cols = cols;
		this->rows = rows;
		values = new int[cols * rows];
	}
};

typedef char* uchar;

void test_matrix();
void print_matrix(matrix* matr);
void print_matrix(double** m, int w, int h);
void print_matrix(int** m, int w, int h);
void print_vector(double* v, int len);
void print_vector(int* v, int len);
void print_vector(double* v, int cols, int rows);

int* prepare_matrix(int cols, int rows);
double* calc_adjacency_matrix(int dm1, int dm2, int* c_values, int cols,
		int rows, int max_i, int max_j);
double* calc_signs(double* adj_matr, int cols, int rows, int dm1, int dm2,
		int max_i, int max_j);
double* calc_symmetric_adjacency_matrix(int* pic, int cols, int rows, int dm1,
		int dm2, int max_i, int max_j);

int get_matrix_size_bytes(int cols, int rows);
//
//__global__ void c_dm1_dm2(int i, int j, int cols, int rows, int dm1, int dm2, int* c_values, int* res);
//__global__ void sum_vector(double* v1, double* v2, int len, int stride);
//
//__device__ int q_ij(int m1, int m2, int d_m1, int d_m2, int i, int j, int cols, int rows, int* pic);
//__device__ int f(int m1, int m2, int cols, int rows, int* pic);

#endif /* IMAGE_STATISTIC_H_ */
