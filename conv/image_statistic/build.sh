nvcc -arch=sm_20 -dc image_statistic.cu
nvcc -arch=sm_20 -dlink image_statistic.o -o dlink.o
g++ image_statistic.o dlink.o test.cpp -lcudart -lpng -lboost_system -lboost_filesystem -std=c++11
