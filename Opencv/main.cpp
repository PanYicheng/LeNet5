#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "lenet5.h"
#include <time.h>
#include<fstream>
#include<iostream>
#include<string>
#include<string.h>
#include<iomanip>
#include<stdio.h>
#include<omp.h>
using namespace std;
using namespace cv;
#define FILE_TRAIN_IMAGE		"train-images.idx3-ubyte."
#define FILE_TRAIN_LABEL		"train-labels.idx1-ubyte"
#define MY_TRAIN_IMAGE           "mytrain26.idx3-ubyte"
#define MY_TRAIN_LABEL           "mylabel26.idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images.idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels.idx1-ubyte"
#define LENET_FILE 		"NewModel.dat"
#define LENET_FILE_OLD  "model.dat"
#define COUNT_TRAIN		60000
#define MY_COUNT_TRAIN  26000
#define MY_TEST_COUNT   100      
#define COUNT_TEST		10000
#define FOLDER          "F:\\Projects\\by_class\\by_class\\"
#define SHOWOUTPUT
#define NUMBERRECOGNIZE
//#define SHOWINPUT
//#define SHOWLAYER1
Mat img(Size(400, 400), CV_8UC1, Scalar(0, 0, 0));
Mat img_white = Mat::ones(400, 400, CV_8UC1)*0;
Point prev_pt = { -1,-1 };
struct CharNum
{
	int character;
	int num;
};

void showfeature(Feature * feature)
{
	char window_name[11][10] = { "input","c10","c11","c12","c13","c14","c15" ,"c16","c17","c18","c19"};
	char window_nameofoutput[] = "Output layer";
	
	
	
#ifdef SHOWINPUT
	Mat img0(Size(32, 32), CV_8UC1, Scalar(0, 0, 0));
	int j = 0;
	for (MatIterator_<uchar> i = img0.begin<uchar>(); i != img0.end<uchar>(); i++,j++)
	{
		(*i) = ( int((*((double *)(feature->input[0]) + j)+1 )* 255) ) % 255;
	}
	imshow(window_name[0], img0);
#endif
#ifdef SHOWLAYER1
	for (int i = 1; i <= LAYER1; i++)
	{
		namedWindow(window_name[i], WINDOW_NORMAL);
	}
	Mat img1(Size(28, 28), CV_8UC1, Scalar(0, 0, 0));
	for (int i = 0; i < LAYER1; i++)
	{
		j = 0;
		for (MatIterator_<uchar> k = img1.begin<uchar>(); k != img1.end<uchar>(); k++, j++)
		{
			(*k) = (int(abs((*((double *)(feature->layer1[i]) + j)+1)) * 255)) % 255;
		}
		imshow(window_name[i+1], img1);
	}
#endif
#ifdef SHOWOUTPUT
	namedWindow(window_nameofoutput, WINDOW_NORMAL);
	Mat img2(Size(300, 300), CV_8UC3, Scalar(0, 0, 0));
	double sum = 0;
	for(int i=0;i<OUTPUT;i++)
	{
		sum += feature->output[i];
	}
	for (int i = 0; i < OUTPUT; i++)
	{
		int height = int(abs(feature->output[i]/sum*400));
		line(img2, Point(i * 11 + 1, 290), Point(i * 11 + 1, 290 - height), Scalar(0, 0, 255), 5);
	}
	imshow(window_nameofoutput, img2);
#endif
	waitKey(1);
	return;
}

void showimg(unsigned char(*data)[28][28], unsigned char label[], const int count)
{
	Mat img(28, 28, CV_8UC1);
	char title[10] = "img0";
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
		cout << int(label[i]) << endl;
		imshow(title, img);
		waitKey(0);
	}
	return;

}

void on_mouse(int event, int x, int y, int flags, void*)
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
		line(img, prev_pt, pt, Scalar(255, 255, 255), 16, 8, 0);
		prev_pt = pt;
		imshow("image", img);
	}
}

int img_process(LeNet5 *lenet)
{
	Mat displabel(Size(30,30),CV_8UC3);
	Mat after(Size(28, 28), CV_8UC1);
	//inputimg=imread("00000.png", 0);
	//resize(inputimg, after, after.size(), INTER_AREA);
	namedWindow("image", CV_WINDOW_NORMAL);
	namedWindow("after", CV_WINDOW_NORMAL);
	namedWindow("display labels", CV_WINDOW_NORMAL);
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
				int label=0;
				resize(img, after, after.size(), INTER_AREA);
				imshow("after", after);
				waitKey(1);
				
				unsigned char img_[28][28];
				for (int y = 0; y < 28; y++)
				{
					for (int x = 0; x < 28; x++)
					{
						img_[y][x] = after.at<unsigned char>(y, x);
					}
				}
			
				int result = Predict(lenet, img_, OUTPUT);
				string dispalpha[26] = { "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z" };
				string dispnum[10] = { "0","1","2","3","4","5","6","7","8","9" };
				displabel.setTo(Scalar(100, 0, 0));
#ifdef NUMBERRECOGNIZE
				putText(displabel, dispnum[result], Point(0, 28), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2, LINE_4, false);
#else
				putText(displabel, dispalpha[result], Point(0, 28), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2, LINE_4, false);
#endif
				imshow("display labels", displabel);
				waitKey(1);
				cout << result << endl;
				/*cin >> label;
				for (int i = 0; i < 20; i++)
				{
					Train(lenet, img_, label);
				}*/
				//imwrite("newimg.jpg", img);
			}
		}
	}

	return 1;
}

string getstring(int n)
{
	stringstream newstr;
	newstr << n;
	return newstr.str();
}

int generate_traindata(void)
{
	char magicnumber[4] = {0,0,8,3};
	int image_count = 13000;
	int image_size = 28;
	ofstream imageout;
	imageout.open("mytrain26.idx3-ubyte", ios::binary | ios::out);
	ofstream labelout;
	labelout.open("mylabel26.idx1-ubyte", ios::binary | ios::out);
	imageout.write(magicnumber, sizeof(magicnumber));
	imageout.write((char *)&image_count, sizeof(image_count));
	imageout.write((char *)&image_size, sizeof(image_size));
	imageout.write((char *)&image_size, sizeof(image_size));

	labelout.write(magicnumber, sizeof(magicnumber));
	labelout.write((char *)&image_count, sizeof(image_count));

	string train_name = "\\train\\";
	string file_header = "00000 (";
	string file_tail = ").png";
	string innerfolder;
	string png_name;
	Mat inputimg;
	Mat after(Size(28, 28), CV_8UC1, Scalar(0, 0, 0));
	image *data=(image *)calloc(1,sizeof(image));
	unsigned char label = 0;
	for (int i = 0; i < 26; i++,label++)
	{
		char temp[2] = { (char)i + int('A') ,'\0'};
		string cha(temp);
		innerfolder = (string)FOLDER + cha + train_name +file_header;
		for (int j = 1; j <= 1000; j++)
		{
			png_name = innerfolder + getstring(j)+file_tail;
			inputimg=imread(png_name,0);
			inputimg = inputimg(Range(33, 100), Range(33, 100));
			printf("Read: %2d   %4d\n", i, j);
			resize(inputimg, after, after.size(), 0, 0, INTER_AREA);
			int y=0;
			for (MatIterator_<uchar> x = after.begin<uchar>(); x != after.end<uchar>(); x++,y++)
			{
				(*x) = 255-(*x);
				*((char *)data + y) = (*x);
			}
			imageout.write((char *)data, sizeof(image));
			labelout.write((char *)&label, sizeof(unsigned char));
		}
	}
	imageout.close();
	labelout.close();
	return 0;
}

int generate_traindata5(void)
{
	int i, j;
	char magicnumber[4] = { 0,0,8,3 };
	int image_count = 26000;
	int image_size = 28;
	ofstream imageout;
	imageout.open("mytrain26.idx3-ubyte", ios::binary | ios::out);
	ofstream labelout;
	labelout.open("mylabel26.idx1-ubyte", ios::binary | ios::out);
	imageout.write(magicnumber, sizeof(magicnumber));
	imageout.write((char *)&image_count, sizeof(image_count));
	imageout.write((char *)&image_size, sizeof(image_size));
	imageout.write((char *)&image_size, sizeof(image_size));

	labelout.write(magicnumber, sizeof(magicnumber));
	labelout.write((char *)&image_count, sizeof(image_count));
	
	string folder = "F:\\Projects\\by_class\\by_class";
	string cha[26] = {"\\A\\","\\B\\","\\C\\","\\D\\" ,"\\E\\"  ,"\\F\\"  ,"\\G\\"  ,"\\H\\"  ,"\\I\\"  ,"\\J\\"  ,"\\K\\"  ,"\\L\\"  ,"\\M\\"  ,"\\N\\"  ,"\\O\\"  ,"\\P\\"  ,"\\Q\\" ,"\\R\\" ,"\\S\\"  ,"\\T\\"  ,"\\U\\"  ,"\\V\\"  ,"\\W\\"  ,"\\X\\"  ,"\\Y\\"  ,"\\Z\\" };
	string file_tail = ".png";
	string innerfolder;
	string jpg_name;
	Mat inputimg;
	Mat after(Size(28, 28), CV_8UC1, Scalar(0, 0, 0));
	image *data = (image *)calloc(1, sizeof(image));
	unsigned char label = 0;
	
	CharNum * randfile = (CharNum *)calloc(image_count, sizeof(CharNum));
	CharNum *p = randfile;
	for (i = 0; i < 26; i++)
	{
		for (j = 1; j <= 1000; j++)
		{
			p->character = i;
			p->num = j;
			p++;
		}
	}
	for (i = 0; i < image_count; i++)
	{
		j = ((int)(double(rand()) * 26000/RAND_MAX) )% 26000;
		CharNum temp = randfile[j];
		randfile[j] = randfile[i];
		randfile[i] = temp;
		//cout << i << "   " << j << endl;
	}
	
	namedWindow("after", WINDOW_NORMAL);
 	for (i = 0; i < image_count; i++)
	{
		innerfolder = folder + cha[randfile[i].character];
		string png_name = innerfolder + getstring(randfile[i].num) + file_tail;
		inputimg = imread(png_name, 0);
		printf("Read: %2d   %4d  ", randfile[i].character, randfile[i].num);
		resize(inputimg, after, after.size(), 0, 0, INTER_AREA);
		int y = 0;
		for (MatIterator_<uchar> x = after.begin<uchar>(); x != after.end<uchar>(); x++, y++)
		{
			(*x) = 255 - (*x);
			*((char *)data + y) = (*x);
		}
		//imshow("after", after);
		//waitKey(1);
		label = (uchar)(randfile[i].character +0);
		printf("Label:   %2d\n", label);
		imageout.write((char *)data, sizeof(image));
		labelout.write((char *)&label, sizeof(unsigned char));

	}
	imageout.close();
	labelout.close();
	return 0;
}

int generate_traindata26(void)
{
	int i, j;
	char magicnumber[4] = { 0,0,8,3 };
	int image_count = 62400;
	int image_size = 28;
	ofstream imageout;
	imageout.open("mytrain26.idx3-ubyte", ios::binary | ios::out);
	ofstream labelout;
	labelout.open("mylabel26.idx1-ubyte", ios::binary | ios::out);
	imageout.write(magicnumber, sizeof(magicnumber));
	imageout.write((char *)&image_count, sizeof(image_count));
	imageout.write((char *)&image_size, sizeof(image_size));
	imageout.write((char *)&image_size, sizeof(image_size));

	labelout.write(magicnumber, sizeof(magicnumber));
	labelout.write((char *)&image_count, sizeof(image_count));

	string folder = "F:\\Projects\\by_class\\by_class";
	string cha[26] = { "\\A\\","\\B\\","\\C\\","\\D\\" ,"\\E\\"  ,"\\F\\"  ,"\\G\\"  ,"\\H\\"  ,"\\I\\"  ,"\\J\\"  ,"\\K\\"  ,"\\L\\"  ,"\\M\\"  ,"\\N\\"  ,"\\O\\"  ,"\\P\\"  ,"\\Q\\" ,"\\R\\" ,"\\S\\"  ,"\\T\\"  ,"\\U\\"  ,"\\V\\"  ,"\\W\\"  ,"\\X\\"  ,"\\Y\\"  ,"\\Z\\" };
	string file_head = "train\\00000 (";
	string file_tail = ").png";
	string innerfolder;
	string jpg_name;
	Mat inputimg;
	Mat after(Size(28, 28), CV_8UC1, Scalar(0, 0, 0));
	image *data = (image *)calloc(1, sizeof(image));
	unsigned char label = 0;

	CharNum * randfile = (CharNum *)calloc(image_count, sizeof(CharNum));
	CharNum *p = randfile;
	for (i = 0; i < 26; i++)
	{
		for (j = 1; j <= 2400; j++)
		{
			p->character = i;
			p->num = j;
			p++;
		}
	}
	for (i = 0; i < image_count; i++)
	{
		j = ((int)(double(rand()) * 62400 / RAND_MAX)) % 62400;
		CharNum temp = randfile[j];
		randfile[j] = randfile[i];
		randfile[i] = temp;
		//cout << i << "   " << j << endl;
	}

	namedWindow("after", WINDOW_NORMAL);
	for (i = 0; i < image_count; i++)
	{
		innerfolder = folder + cha[randfile[i].character];
		string png_name = innerfolder + file_head+getstring(randfile[i].num) + file_tail;
		inputimg = imread(png_name, 0); 
		inputimg = inputimg(Range(33, 100), Range(33, 100));
		//printf("Read: %2d   %4d  ", randfile[i].character, randfile[i].num);
		resize(inputimg, after, after.size(), 0, 0, INTER_AREA);
		int y = 0;
		for (MatIterator_<uchar> x = after.begin<uchar>(); x != after.end<uchar>(); x++, y++)
		{
			(*x) = 255 - (*x);
			*((char *)data + y) = (*x);
		}
		//imshow("after", after);
		//waitKey(1);
		label = (uchar)(randfile[i].character + 0);
		//printf("Label:   %2d\n", label);
		imageout.write((char *)data, sizeof(image));
		labelout.write((char *)&label, sizeof(unsigned char));

	}
	imageout.close();
	labelout.close();
	return 0;
}

int generate_testdata(void)
{
	namedWindow("after Cut", WINDOW_NORMAL);
	namedWindow("after", WINDOW_NORMAL);
	char magicnumber[4] = { 0,0,8,3 };
	int image_count = 100;
	int image_size = 28;
	ofstream imageout;
	imageout.open("mytest.idx3-ubyte", ios::binary | ios::out);
	ofstream labelout;
	labelout.open("mytest.idx1-ubyte", ios::binary | ios::out);
	imageout.write(magicnumber, sizeof(magicnumber));
	imageout.write((char *)&image_count, sizeof(image_count));
	imageout.write((char *)&image_size, sizeof(image_size));
	imageout.write((char *)&image_size, sizeof(image_size));

	labelout.write(magicnumber, sizeof(magicnumber));
	labelout.write((char *)&image_count, sizeof(image_count));
	string folder = "F:\\Projects\\by_class\\by_class\\test\\";
	string file_header = "0 (";
	string file_tail = ").png";
	string file_name;
	Mat inputimg;
	Mat aftercut;
	Mat after(28, 28, CV_8UC1,Scalar(0,0,0));
	image *data = (image *)calloc(1, sizeof(image));
	unsigned char label = 10;
	int label_cin = 10;
	for (int i = 1; i <= 100; i++)
	{
		file_name = folder + file_header + getstring(i) + file_tail;
		inputimg = imread(file_name,IMREAD_GRAYSCALE);
		aftercut = inputimg(Range(33, 100), Range(33, 100));
		imshow("after Cut", aftercut);
		resize(aftercut, after, after.size(), 0, 0, INTER_AREA);
		imshow("after", after);
		waitKey(10);
		int y = 0;
		for (MatIterator_<uchar> x = after.begin<uchar>(); x != after.end<uchar>(); x++, y++)
		{
			(*x) = 255 - (*x);
			*((char *)data + y) = (*x);
		}
		imageout.write((char *)data, sizeof(image));
		cin >> label_cin;
		label = (uchar)label_cin;
		cout << endl << "Scanned: " << (char)((int)label-10+int('A'))<<endl;
		labelout.write((char *)&label, sizeof(unsigned char));
	}
	imageout.close();
	labelout.close();
	return 0;
}

void my_tranform(void)
{
	string folder = "F:\\Projects\\by_class\\train";
	string cha[5] = { "\\C\\","\\D\\","\\M\\","\\H\\","\\S\\" };
	string file_header = "0 (";
	string file_tail = ").png";
	Mat inputimg;
	Mat aftercut;
	Mat after(28, 28, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 0; i < 5; i++)
	{
		string innerfolder = folder + cha[i];
		for (int j = 1; j <= 1000; j++)
		{
			string file_name = innerfolder +file_header + getstring(j) + file_tail;
			string write_filename = innerfolder + getstring(j) + ".jpg";
			inputimg = imread(file_name, IMREAD_GRAYSCALE);
			aftercut = inputimg(Range(33, 100), Range(33, 100));
			imshow("after Cut", aftercut);
			resize(aftercut, after, after.size(), 0, 0, INTER_AREA);
			imshow("after", after);
			waitKey(10);
			vector<int> compression_params;
			compression_params.push_back(IMWRITE_JPEG_QUALITY);
			imwrite(write_filename,after);
		}
	}

}

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
	fstream image;
	fstream file_label;
	image.open(data_file,ios::in |ios::binary);
	file_label.open(label_file, ios::in | ios::binary);
	// FILE *fp_label = fopen(label_file, "rb");
	//if (!fp_image||!fp_label) return 1;
	//fseek(fp_image, 16, SEEK_SET);
	//fseek(fp_label, 8, SEEK_SET);
	image.seekg(16, ios::beg);
	file_label.seekg(8, ios::beg);
	//fread(data, sizeof(*data)*count, 1, fp_image);
	//fread(label,count, 1, fp_label);
	image.read((char *)data, sizeof(*data)*count);
	file_label.read((char *)label, count);
	//showimg(data, label, 10);
	//showinputimg(data, label,10);
	image.close();
	file_label.close();
	//fclose(fp_image);
	//fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	//total_size - batch_size
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], OUTPUT);
		//printf("%2d     %2d \n", l, p);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}

int save(LeNet5 *lenet, char filename[])
{
	//FILE *fp = fopen(filename, "wb");
	fstream fp;
	fp.open(filename, ios::binary | ios::out);
	if (!fp) return 1;
	fp.write((char *)lenet, sizeof(LeNet5));
	fp.close();
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	//FILE *fp = fopen(filename, "rb");
	fstream fp;
	fp.open(filename, ios::in | ios::binary);
	if (!fp) return 1;
	fp.read((char *)lenet, sizeof(LeNet5));
    fp.close();
	return 0;
}



void foo()
{
#ifdef STARTTRAIN
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	image *my_traindata= (image *)calloc(MY_COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	uint8 *my_label = (uint8 *)calloc(MY_COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(my_traindata, my_label, MY_COUNT_TRAIN, MY_TRAIN_IMAGE, MY_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(my_traindata);
		free(my_label);
		system("pause");
	}
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{

		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}
#endif
	//showimg(train_data, train_label, 200);

	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    char cha_modelname[] = "NewModel_char26_feature10.dat";
	char num_modelname[] = "model.dat";


	if (load(lenet, num_modelname))
	{
		Initial(lenet); printf("Failed to Read model from file,initial LeNet\n");
	}
	else {
		printf("Succed Read LeNet model\n");
	}
		//clock_t start = clock();
	int batches[] = { 300 };
   //for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
	//training(lenet, train_data, train_label,300, 60000);
	//training(lenet, my_traindata, my_label,100, 26000);
	//save(lenet, "NewModel_char26_feature10.dat");
	//int right = testing(lenet, test_data, test_label, COUNT_TEST);
	//printf("%d/%d\n", right, COUNT_TEST);
	//printf("Time:%u\n", (unsigned)(clock() - start));
	//int right = testing(lenet, my_traindata, my_label, 10000);
	//printf("%d/%d\n", right, 10000);
	
	
	img_process(lenet);
	free(lenet);
#ifdef STARTTRAIN
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
#endif
	//system("pause");
}

int main()
{
	//my_tranform();
	//generate_traindata5();
	//generate_testdata();
	//generate_traindata();
	//generate_traindata26();
	//printf("finish generate\n");
	foo();
	return 0;
}