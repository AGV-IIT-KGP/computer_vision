#include<iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat thresholding(Mat &aimage,int trl,int trh,int tbl,int tbh,int tgl,int tgh){
 
  Mat hsv;
  cvtColor(aimage,hsv,CV_BGR2HSV);
  int num_rows=hsv.rows;
  int num_cols=hsv.cols;
  
 
  for(int i=0;i<num_rows;i++)
  {
	 for(int j=0;j<num_cols;j++)
	  {
	 	 if(hsv.at<Vec3b>(i,j)[0]>trl && hsv.at<Vec3b>(i,j)[0]<trh)
			 {
				 hsv.at<Vec3b>(i,j)[0]=0;
				 hsv.at<Vec3b>(i,j)[1]=100;
				 hsv.at<Vec3b>(i,j)[2]=0;
			 }
		 else if(hsv.at<Vec3b>(i,j)[0]>tbl && hsv.at<Vec3b>(i,j)[0]<tbh)
			 {
				 hsv.at<Vec3b>(i,j)[0]=0;
				 hsv.at<Vec3b>(i,j)[1]=100;
				 hsv.at<Vec3b>(i,j)[2]=0;
			 }
		else if(hsv.at<Vec3b>(i,j)[0]>tgl && hsv.at<Vec3b>(i,j)[0]<tgh)
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
  
  namedWindow("My Window",1);

  int iSliderValue1 = 0;
     createTrackbar("Red_Low", "My Window", &iSliderValue1, 179);
  int iSliderValue2 = 16;
     createTrackbar("Red_High", "My Window", &iSliderValue2, 179);
  int iSliderValue3 = 103;
     createTrackbar("Blue_Low", "My Window", &iSliderValue3, 179);
  int iSliderValue4 = 122;
     createTrackbar("Blue_High", "My Window", &iSliderValue4, 179);  
  int iSliderValue5 = 163;
     createTrackbar("Brown_Low", "My Window", &iSliderValue5, 179);
  int iSliderValue6 = 179;
     createTrackbar("Brown_High", "My Window", &iSliderValue6, 179);  
 
  namedWindow("Display Thresh", CV_WINDOW_AUTOSIZE);

  while(true){
  t=thresholding(image,iSliderValue1,iSliderValue2,iSliderValue3,iSliderValue4,iSliderValue5,iSliderValue6);
  
  imshow("Display Thresh",t);

  int iKey = waitKey(50);

          //if user press 'ESC' key
          if (iKey == 27)
          {
               break;
          } 

  }
  return 0;

}
