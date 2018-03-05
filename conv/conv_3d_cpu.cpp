/*
+ * conv_2d_cpu.cpp
 *
 *  Created on: Mar 5, 2018
 *      Author: lyan
 */

int*** pad(int*** input, int pad){

}

void conv(int*** inp, int*** ker, int*** out, int i, int j, int k, int kCols, int kRows){

}

int*** conv_3d(int*** in, int*** kernel, int cols,int rows, int d, int kCols,int kRows,int kd, int stride){
	int kCenterX = kCols / 2;
	int kCenterY = kRows / 2;

	int mm = 0;
	int nn = 0;
	int ii = 0;
	int jj = 0;

	for(i=0; i < rows; i += stride)              // rows
	{
	    for(j=0; j < cols; j += stride)          // columns
	    {
	        for(m=0; m < kRows; m ++)     // kernel rows
	        {
	            mm = kRows - 1 - m;      // row index of flipped kernel

	            for(n=0; n < kCols; ++n) // kernel columns
	            {
	                nn = kCols - 1 - n;  // column index of flipped kernel

	                // index of input signal, used for checking boundary
	                ii = i + (m - kCenterY);
	                jj = j + (n - kCenterX);

	                // ignore input samples which are out of bound
	                if( ii >= 0 && ii < rows && jj >= 0 && jj < cols )
	                    out[i][j] += in[ii][jj] * kernel[mm][nn];
	            }
	        }
	    }
	}
}

int main(char** argv, int argc){

}
