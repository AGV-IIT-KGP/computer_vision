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
	Mat img =imread( "/home/harshit/Desktop/noise2.png", CV_LOAD_IMAGE_GRAYSCALE );
	//imshow("original", img);
	waitKey(1);


	/*int erosion_size = 3;  
       	Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size) );
	erode(img,img,element);
	dilate(img,img,element);
	imshow("eroded and diluted", img);
	waitKey(3000);*/                              //erosion and dilution

	medianBlur ( img, img, 15 );
	imshow("median blurred", img);
	//waitKey(1);

	Mat copy=img.clone();

	vector<co_ord> points;
	
	//extract white pixels in vector points
	int i, j, k;
	for(i=0;i<img.rows;i++)
		for(j=0;j<img.cols;j++)
		{
			if(img.at<uchar>(i,j)>=20) points.push_back({j,i});
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
	float dir[2];
	dir[0]=best_c;

	printf("\nNumber of matches(first lane): %d\n", best_matches);

	//mark the points which lie on the best curve and remove these points from original image
	Mat output1(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));
	for(i=0;i<points.size();i++)
		if(on_curve(points[i],best_a,best_b,best_c)==true)
		{
			output1.at<Vec3b>(points[i].y, points[i].x)={255,255,0};
			copy.at<uchar>(points[i].y,points[i].x)=0;
		}
	//imshow("single lane", copy);
	//imshow("first lane", output1);


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
	dir[1]=best_c;

	printf("Number of matches(second lane): %d\n", best_matches);

	//mark the points which lie on the best curve
	Mat output2(copy.rows,copy.cols,CV_8UC3,Scalar(0,0,0));
	for(i=0;i<points.size();i++)
		if(on_curve(points[i],best_a,best_b,best_c)==true)
		{
			output2.at<Vec3b>(points[i].y, points[i].x)={255,255,0};
		}
	
	if(dir[0]>dir[1]) 
	{
		imshow("right lane", output1);
		imshow("left lane",output2);
		waitKey(10000000);
	}
	else
	{

		imshow("right lane", output2);
		imshow("left lane",output1);
		waitKey(1000000);	
	}

	Mat output(copy.rows,copy.cols,CV_8UC3,Scalar(0,0,0));
	for(i=0;i<output.rows;i++)
		for(j=0;j<output.cols;j++)
			if(output1.at<Vec3b>(i,j)[0]==255 || output2.at<Vec3b>(i,j)[0]==255) output.at<Vec3b>(i,j)={0,0,255};
	//imshow("Lanes", output);
	imwrite("../images/lanes.jpg", output);
	waitKey(0);


	
	
	return 0;
	
}
