#include<stdio.h>
#include<stdlib.h>
#include"opencv/cv.h"
#include<opencv2/highgui/highgui.hpp>
#include<bits/stdc++.h>
#include<typeinfo>
#include<iostream>
#include<vector>
#include<time.h>
#include<algorithm>
#include<cmath>

using namespace cv;
using namespace std;

struct co_ord_ {int x;int y;};
typedef struct co_ord_ co_ord;

void compute_quadratic(co_ord p1, co_ord p2, co_ord p3, float& a, float& b, float &c)
{
	if(p1.y==p2.y || p1.y==p3.y)	//error check for denominator in expression below
	{
		a=0;
		b=0;
		c=0;
		return;
	}

	
	a=( ((float)p1.x-p2.x)/(p1.y-p2.y)- ((float)p2.x-p3.x)/(p2.y-p3.y) )/((float)p1.y-p3.y);
	b=( ((float)p1.x-p2.x)- a*(p1.y*p1.y- p2.y*p2.y) )/((float)p1.y-p2.y);
	c=p1.x-a*p1.y*p1.y-b*p1.y;
	return ;
		
}

bool on_curve(co_ord p, float a, float b, float c) //checks if given point lies on curve
{
	float error_margin=25;
	if(abs(a*p.y*p.y+b*p.y+c-p.x)<error_margin) return true;
	else return false;
}

int main()
{
	Mat img =imread( "../images/noise.png", CV_LOAD_IMAGE_GRAYSCALE );
	imshow("original", img);
	waitKey(1);


	/*int erosion_size = 3;  
       	Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size) );
	erode(img,img,element);
	dilate(img,img,element);

	imshow("eroded and diluted", img);
	waitKey(3000);*/                              //erosion and dilution

	medianBlur ( img, img, 15 );
	//imshow("median blurred", img);

	Mat copy=img.clone();

	vector<co_ord> points;
	
	//extract white pixels in vector points
	int i, j, k;
	for(i=0;i<img.rows;i++)
		for(j=0;j<img.cols;j++)
		{
			if(img.at<uchar>(i,j)>=20) points.push_back({j,i});
		}
	if(points.size()==0)	//no white pixel in image
	{
		cout<<"No white pixel in image"<<endl;
		return -1;
	}


	float a, b, c;
	float best_a=0, best_b=0, best_c=0;
	int matches, best_matches=0;

	//run ransac to fit a quadratic model
	co_ord p1, p2, p3, best_p;
	int iterations=100;
	srand(12);
	int tag;
	while(iterations--)	
	{
		//randomly select three distinct points
		tag=1;
		do
		{
			p1=points[rand()%points.size()];
			p2=points[rand()%points.size()];
			p3=points[rand()%points.size()];
		}
		while(p1.y==p2.y || p1.y==p3.y);
		//compute curve corresponding to this three points
		compute_quadratic(p1,p2,p3,a,b,c);
		//find number of points lying on this curve
		matches=0;
		for(i=0;i<points.size();i++)
			if(on_curve(points[i], a, b, c)==true) matches++;
		//if curve is better than current best, update current best
		if(matches>=best_matches)
		{
			best_matches=matches;
			best_a=a;
			best_b=b;
			best_c=c;
		}
	}

	printf("\nNumber of matches(first lane): %d\n", best_matches);

	//mark the points which lie on the best curve and remove these points from original image
	Mat output_first_lane(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));
	for(i=0;i<points.size();i++)
		if(on_curve(points[i],best_a,best_b,best_c)==true)
		{
			output_first_lane.at<Vec3b>(points[i].y, points[i].x)={255,255,0};
			copy.at<uchar>(points[i].y,points[i].x)=0;
		}
	//imshow("single lane", copy);
	imshow("first lane", output_first_lane);


	//
	//ransac for second lane
	//
	points.clear();
	for(i=0;i<copy.rows;i++)
		for(j=0;j<copy.cols;j++)
		{
			if(copy.at<uchar>(i,j)>=20) points.push_back({j,i});
		}


	best_a=0;
	best_b=0;
	best_c=0;
	best_matches=0;

	iterations=100;
	srand(12);
	while(iterations--)	
	{
		//randomly select three distinct points
		tag=1;
		do
		{
			p1=points[rand()%points.size()];
			p2=points[rand()%points.size()];
			p3=points[rand()%points.size()];
		}
		while(p1.y==p2.y || p1.y==p3.y);
		//compute curve corresponding to this three points
		compute_quadratic(p1,p2,p3,a,b,c);
		//find number of points lying on this curve
		matches=0;
		for(i=0;i<points.size();i++)
			if(on_curve(points[i], a, b, c)==true) matches++;
		//if curve is better than current best, update current best
		if(matches>=best_matches)
		{
			best_matches=matches;
			best_a=a;
			best_b=b;
			best_c=c;
		}
	}

	printf("Number of matches(second lane): %d\n", best_matches);

	int threshold=2000;
	if(best_matches<=threshold)	//pixels detected in second lane below certain threshold i.e. only one lane in image
	{
		Mat output=output_first_lane;
		imshow("Lanes", output);
		imwrite("../images/lanes.jpg", output);
		waitKey(0);
		return 0;
	}

	//mark the points which lie on the best curve
	Mat output_second_lane(copy.rows,copy.cols,CV_8UC3,Scalar(0,0,0));
	for(i=0;i<points.size();i++)
		if(on_curve(points[i],best_a,best_b,best_c)==true)
		{
			output_second_lane.at<Vec3b>(points[i].y, points[i].x)={255,255,0};
		}
	
	imshow("second lane", output_second_lane);

	Mat output(copy.rows,copy.cols,CV_8UC3,Scalar(0,0,0));
	for(i=0;i<output.rows;i++)
		for(j=0;j<output.cols;j++)
			if(output_first_lane.at<Vec3b>(i,j)[0]==255 || output_second_lane.at<Vec3b>(i,j)[0]==255) output.at<Vec3b>(i,j)={0,0,255};
	imshow("Lanes", output);
	imwrite("../images/lanes.jpg", output);
	waitKey(0);


	
	
	return 0;
	
}
