//#define cimg_display 0
//#include <png++/png.hpp>
#include <iostream>
//#include "image_statistic.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>


using namespace std;
using namespace cv;

const static char* image_source_path =
		"/media/dodler/CEE4E048E4E033FD/samples/KylbergTextureDataset-1.0-png-originals/blanket2-a.png";

static const int CLUSTER_NUM = 20;

int main(void){
//    Mat img = imread("/home/lyan/Downloads/KylbergTextureDataset-1.0-png-originals.7z.001.1/KylbergTextureDataset-1.0-png-originals/blanket2-a.png");
    Mat img = imread("//home/lyan/Pictures/Lenna.png");

    cv::Mat labels, data;
    cv::Mat centers(CLUSTER_NUM, 1, CV_32FC1);
    img.convertTo(data, CV_32F);

//    kmeans(data, 10,labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),3, KMEANS_PP_CENTERS, &centers);
    double comp = kmeans(data, CLUSTER_NUM, labels,
                TermCriteria(CV_TERMCRIT_ITER, 10, 1.0),
                10, KMEANS_PP_CENTERS, centers);
    imshow("result", data);
    data.convertTo(data, CV_32FC3);

    cout << centers << endl;
    cout << comp << endl;

    waitKey();
    return 0;
}
