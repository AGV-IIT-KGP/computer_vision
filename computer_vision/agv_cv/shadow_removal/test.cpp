#include "shadowRemoval.h"

int main(int argc, char* argv[]){
  
  cv::VideoCapture capture(argv[1]);
  if (!capture.isOpened()){
      cv::VideoCapture capture(argv[1]);
  }
  cv::Mat original;
  cv::namedWindow("output",1);
  while(1){
    capture >> original;
    cv::namedWindow("original_video",1);
    cv::imshow("original_video",original);
    cv::imwrite("final_output.jpg",original);
    cv::waitKey(20);
    while(original.empty()){
      capture >> original;
    }
    cv::Mat final_image(original.rows,original.cols,CV_8UC3,cvScalarAll(0));
    if (argc == 3){
      ShadowRemoval Shadow(original,atoi(argv[2]));
    	final_image = Shadow.shadowRemoval();
    }
    else{
    	ShadowRemoval Shadow(original,0);
    	final_image = Shadow.shadowRemoval();
    }
    cv::imshow("output",final_image);
    cv::waitKey(20);
  }
}