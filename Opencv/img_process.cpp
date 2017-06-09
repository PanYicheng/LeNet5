#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include"Lenet5.h"
using namespace std;
using namespace cv;
Mat img(Size(28,28),CV_8UC1,Scalar(0,0,0));
Mat img_white = Mat::zeros(28, 28, CV_8UC1);
Point prev_pt = { -1,-1 };

void showimg(unsigned char(*data)[28][28], unsigned char label[], const int count)
{
	Mat img(28, 28,CV_8UC1);
	char title[10] =  "img0" ;
	namedWindow(title, CV_WINDOW_NORMAL);
	for (int i = 0; i < count; i++)
	{
		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				img.at<unsigned char>(y, x) = data[i][y][x];
			}
		}
		imshow(title, img);
		waitKey(0);
	}
	return;

}
void on_mouse(int event, int x, int y, int flags, void* )
{
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON))
		prev_pt = Point(-1, -1);
	else if (event == CV_EVENT_LBUTTONDOWN)
		prev_pt = Point(x, y);
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		Point pt = Point(x, y);
		if (prev_pt.x < 0)
			prev_pt = pt;
		line(img, prev_pt, pt, Scalar(255, 255, 255), 5, 4, 0);
		prev_pt = pt;
		imshow("image", img);
	}
}


int img_process(LeNet5 *lenet)
{
	namedWindow("image", 1);
	imshow("image", img);
	setMouseCallback("image", on_mouse, 0);
	int oldresult = 0;
	for (;;)
	{
		int c = cvWaitKey(0);

		if ((char)c == 27)
			break;

		if ((char)c == 'r')
		{
			img_white.copyTo(img);
			imshow("image", img);
		}
		if ((char)c == 's' || (char)c == '\n') {
			if ((char)c == 's')
			{
				//加入降低大小的代码

				imwrite("newimg.jpg", img);
			}
		}
		unsigned char img_[28][28];
		for (int y = 0; y < 28; y++)
		{
			for (int x = 0; x < 28; x++)
			{
				img_[y][x] = img.at<unsigned char>(y, x);
			}
		}
		int result = Predict(lenet, img_, 10);
		if (result != oldresult)
		{
			cout << result << endl;
			oldresult = result;
		}
	}

	return 1;
}