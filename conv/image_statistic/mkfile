all: program

program: image_statistic.o
	g++ -o program -L/usr/local/cuda/lib64  -lpng test.cpp  image_statistic.o

image_statistic.o:
	nvcc -c -arch=sm_20 image_statistic.cu 

clean: rm -rf *o program
