#include<iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat thresholding(Mat &aimage){
 
  Mat hsv;
  cvtColor(aimage,hsv,CV_BGR2HSV);
  int num_rows=hsv.rows;
  int num_cols=hsv.cols;
  
 
  for(int i=0;i<num_rows;i++)
  {
	 for(int j=0;j<num_cols;j++)
	  {
	 	 if(hsv.at<Vec3b>(i,j)[0]>0 && hsv.at<Vec3b>(i,j)[0]<10)
			 {
				 hsv.at<Vec3b>(i,j)[0]=0;
				 hsv.at<Vec3b>(i,j)[1]=100;
				 hsv.at<Vec3b>(i,j)[2]=0;
			 }
		 else if(hsv.at<Vec3b>(i,j)[0]>100 && hsv.at<Vec3b>(i,j)[0]<200)
			 {
				 hsv.at<Vec3b>(i,j)[0]=0;
				 hsv.at<Vec3b>(i,j)[1]=100;
				 hsv.at<Vec3b>(i,j)[2]=0;
			 }
		 else
			 {
				 hsv.at<Vec3b>(i,j)[0]=0;
				 hsv.at<Vec3b>(i,j)[1]=0;
				 hsv.at<Vec3b>(i,j)[2]=0;
			 }

	 }
 
  }
 
  return hsv;
}

int main(){
  Mat image,t;
 
  image=imread("test_flags.jpg");
  namedWindow("Display Original", CV_WINDOW_AUTOSIZE);
  imshow("Display Original",image);
 
  t=thresholding(image);
  namedWindow("Display Thresh", CV_WINDOW_AUTOSIZE);
  imshow("Display Thresh",t);
 
  waitKey(0);
  return 0;

}
