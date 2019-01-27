//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include<opencv2/dnn.hpp>

#include <iostream>
#include <algorithm> 
#include <time.h>
//#include <resource.h>
#ifndef __EXTRACT_TABLE_HPP__
#define __EXTRACT_TABLE_HPP__
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#endif
using namespace cv;
using namespace std;
Mat yy(Mat src1, Mat src2, Mat dst);
Mat yy2(Mat src1, Mat src2, Mat dst);
void AdaptiveFindThreshold(cv::Mat src, double *low, double *high, int aperture_size = 3);
void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high);
Mat extractTable(Mat img);
void opencvpb();
#define V_PROJECT 1
#define H_PROJECT 2
Mat canny;
Mat gray, sobel, edge, erod, blur;
int nHeight;//panlin
typedef struct a
{
	int line;
	int x;
	int y;
	int width;
	int height;
	int NUM;
};
struct a tid[300];

typedef struct
{
	int begin;
	int end;

}char_range_t;

void draw_projection(vector<int>& pos, int mode)
{
	vector<int>::iterator max = std::max_element(std::begin(pos), std::end(pos)); //求最大值
	if (mode == H_PROJECT)
	{
		int height = pos.size();
		int width = *max;
		Mat project = Mat::zeros(height, width, CV_8UC1);
		for (int i = 0; i < project.rows; i++)
		{
			for (int j = 0; j < pos[i]; j++)
			{
				project.at<uchar>(i, j) = 255;                //project.at<uchar>(i, j)表示i行j列的这个像素
			}
		}
		//cvNamedWindow("水平投影", 0);
		//imshow("水平投影", project);                 //*********************水平投影
		imwrite("D:\\test\\水平投影.jpg", project);
	}
	else if (mode == V_PROJECT)
	{
		int height = *max;
		int width = pos.size();
		Mat project = Mat::zeros(height, width, CV_8UC1);
		for (int i = 0; i < project.cols; i++)
		{
			for (int j = project.rows - 1; j >= project.rows - pos[i]; j--)
			{
				//std::cout << "j:" << j << "i:" << i << std::endl;
				project.at<uchar>(j, i) = 255;
			}
		}
		//cvNamedWindow("垂直投影", 0);
		//imshow("垂直投影", project);
		imwrite("D:\\test\\垂直投影.jpg", project);
	}

	//waitKey();
}

//获取文本的投影用于分割字符(垂直，水平)
int GetTextProjection(Mat &src, vector<int>& pos, int mode)             //raw line
{
	if (mode == V_PROJECT)
	{
		for (int i = 0; i < src.rows; i++)                  
		{
			uchar* p = src.ptr<uchar>(i);           //第i行的头指针
			for (int j = 0; j < src.cols; j++)
			{
				if (p[j] == 0)
				{
					pos[j]++;
				}
			}
		}

		draw_projection(pos, V_PROJECT);                //垂直投影
	}
	else if (mode == H_PROJECT)
	{
		for (int i = 0; i < src.cols; i++)
		{

			for (int j = 0; j < src.rows; j++)
			{
				if (src.at<uchar>(j, i) == 0)
				{
					pos[j]++;
				}
			}
		}
		draw_projection(pos, H_PROJECT);         //

	}

	return 0;
}

//获取每个分割字符的范围，min_thresh：波峰的最小幅度，min_range：两个波峰的最小间隔  
vector<int> GetPeekRange(vector<int> &vertical_pos, vector<char_range_t> &peek_range, float min_thresh, int min_range)
//int GetPeekRange(vector<int> &vertical_pos, vector<char_range_t> &peek_range, float min_thresh, int min_range)
{
	int begin = 0;
	int end = 0;

	vector<int> label;
	for (int i = 0; i < vertical_pos.size(); i++)
	{

		if (vertical_pos[i] > min_thresh && begin == 0)
		{
			begin = i;
			int y = begin;   //lfs
			label.push_back(y);   //lfs
			
		}
		else if (vertical_pos[i] > min_thresh && begin != 0)
		{
			continue;
			//return 0;
		}
		else if (vertical_pos[i] < min_thresh && begin != 0)
		{
			end = i;
			if (end - begin >= min_range)
			{
				char_range_t tmp;
				tmp.begin = begin;
				tmp.end = end;
				peek_range.push_back(tmp);
				begin = 0;
				end = 0;
			}
			//return begin;      //lfs

		}
		else if (vertical_pos[i] < min_thresh || begin == 0)
		{
			continue;
		}
		else
		{
			//printf("raise error!\n");
		}
	}

	return label;  //lfs   返回每行的y坐标
	//return 0;    
}

inline void save_cut(const Mat& img, int id)
{
	char name[128] = { 0 };
	sprintf(name, "D:\\test\\test\\%d.jpg", id);
	imwrite(name, img);
}


typedef struct Array
{
	int NUM;   //每行字符的总个数
	vector<vector<int>> matrices;  //矩阵信息
}Array;

//框出字符
Array CutChar(Mat &img, const vector<char_range_t>& v_peek_range, const vector<char_range_t>& h_peek_range, vector<Mat>& chars_set, int line_num)
{
	Array Char;
	vector<vector<int> > Matrix(50, vector<int>(3));
	int count = 0;
	Mat show_img = img.clone();      //img.clone()是深拷贝，show_img是img的副本。拷贝的图片和原图完全一样，后面的操作也不会对其产生影响  
	cvtColor(show_img, show_img, CV_GRAY2RGB);
	//namedWindow("GaussianBlur", WINDOW_NORMAL);
	//imshow("GaussianBlur", show_img);
	for (int i = 0; i < v_peek_range.size(); i++)
	{
		int char_gap = v_peek_range[i].end - v_peek_range[i].begin;
		{
			int x = v_peek_range[i].begin - 2>0 ? v_peek_range[i].begin - 2 : 0;
			int width = char_gap + 4 <= img.rows ? char_gap : img.rows;
			Rect r(x, 0, width, img.rows);                  //(参数1：左上角的x坐标；参数2：左上角y坐标；参数3：矩形宽；参数4：矩形高)
			rectangle(show_img, r, Scalar(255, 0, 0), 1);
			//Rectangle( CvArr* img, CvPoint pt1矩形顶点, CvPoint pt2矩形顶点, CvScalar color线条颜色, int thickness=1线条粗细, int line_type=8, int shift=0 ); 
			Mat single_char = img(r).clone();
			chars_set.push_back(single_char);
			save_cut(single_char, count);

			//Matrix[count][0] = x;              //矩阵数据没保留下来？？？？为什么呢
			Matrix[count][1] = width;
			Matrix[count][2] = img.rows;
			Matrix[count][0] = x;
			//Matrix[count][1] = single_char.cols;
			//Matrix[count][2] = single_char.rows;
			count++;
		}
	}
	printf("%d", count);
	//imshow("cut", show_img);
	imwrite("D:\\test\\cut.jpg", show_img);
	waitKey();        // 一行行显示代码
	printf("count = %d", count);

	Char.NUM = count;
	Char.matrices = Matrix;

	return Char;
}

Mat cut_one_line(const Mat& src, int begin, int end)
{
	Mat line = src(Rect(0, begin, src.cols, end - begin)).clone();
	return line;
}

vector<Mat> CutSingleChar(Mat& img)
{
	clock_t clock_start, clock_end;
	Mat show = img.clone();
	cvtColor(show, show, CV_GRAY2BGR);
	threshold(img, img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//imshow("binary", img);       
	imwrite("D:\\test\\binary.jpg", img);
	//*******************binary 二值化图
	vector<int> horizion_pos(img.rows, 0);
	vector<char_range_t> h_peek_range;
	GetTextProjection(img, horizion_pos, H_PROJECT);              //获取完整图水平投影
	//int h = img.rows / 7;

	//GetPeekRange(horizion_pos, h_peek_range, 8, 40);              //水平分割  panlin
	vector<int>y_label = GetPeekRange(horizion_pos, h_peek_range, 8, 40);          //lfs
	
	//for (int i = 0; i < img.rows; i++)         //lfs
	//{
	//	printf("%d :%d  \n", i , horizion_pos[i]);
	//}


#if 1

	/*将每一文本行切割*/
	vector<Mat> lines_set;

	Array cutchar;
	int total_num = 0;

	vector<Mat> chars_set;

	for (int j = 0; j < h_peek_range.size(); j++)
	{
		Mat line2 = cut_one_line(img, h_peek_range[j].begin, h_peek_range[j].end);
		lines_set.push_back(line2);
	}
	for (int i = 0; i < lines_set.size(); i++)
	{
		//	printf(lines_set[0]);


		Mat line = lines_set[i];
		//Mat line2 = lines_set_show[i];
		//imshow("raw line", line);
		imwrite("D:\\test\\rawline.jpg", line);
		vector<int> vertical_pos(line.cols, 0);
		vector<char_range_t> v_peek_range;
		GetTextProjection(line, vertical_pos, V_PROJECT);           //一行行进行垂直投影
		//获取每个分割字符的范围
		GetPeekRange(vertical_pos, v_peek_range, 0.8, 20);                   //垂直分割        panlin 

		clock_start = clock();
		cutchar = CutChar(line, v_peek_range, h_peek_range, chars_set, i);         //这里进行cut操作    一行行显示出来  ********************************************
		//namedWindow("GaussianBlur", WINDOW_NORMAL);
		//imshow("GaussianBlur", line);
		clock_end = clock();

		//typedef struct Array
		//{
		//	int NUM;   //每行字符的总个数
		//	vector<vector<vector<int> > > matrices;  //矩阵信息
		//}Array;
		int numb_count;
		//namedWindow("GaussianBlur", WINDOW_NORMAL);
		//imshow("GaussianBlur", line);
		for (int numb_count = 0; numb_count < cutchar.NUM; numb_count++)
		{
			tid[total_num].x = cutchar.matrices[numb_count][0];
			//tid[total_num].y = i*nHeight + (nHeight - line.rows) / 2;  //panlin
			tid[total_num].y = y_label[i];
		
			tid[total_num].width = cutchar.matrices[numb_count][1];
			tid[total_num].height = cutchar.matrices[numb_count][2] + 5;
			tid[total_num].line = i;
			total_num = total_num + 1;
		}

		//double time = ((double)(clock_end - clock_start) / CLOCKS_PER_SEC);
		//cout << "time:" << time * 1000 << "ms" << endl;

	}

#endif

	return chars_set;
}

////////////////


Mat extractTable(Mat src) {
	double or_src_height = src.cols, or_src_width = src.rows;
	//Mat src = imread("G:\\characters segmentation\\picture\\chart22.jpg",0);
	//resize(src, src, Size(src.cols *0.6, src.rows *0.6), 0, 0, INTER_LINEAR);
	if (!src.data) {
		cout << "not picture" << endl;
	}
	Mat canny, gray, sobel, edge, erod, blur;

	double src_height = src.cols, src_width = src.rows;
	//namedWindow("source", WINDOW_NORMAL);
	//imshow("source", src);

	//先转为灰度图
	cvtColor(src, gray, COLOR_BGR2GRAY);
	//threshold(gray, gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	//imwrite("D:/result/gray.jpg", gray);

	//腐蚀（黑色区域变大）
	int erodeSize = src_height *0.003;// 300;
	if (erodeSize % 2 == 0)
		erodeSize++;
	Mat element = getStructuringElement(MORPH_RECT, Size(erodeSize, erodeSize));
	erode(gray, erod, element);
	imwrite("D:\\test\\erod.jpg", erod);
	//高斯模糊化
	int blurSize = src_height*0.005; // 200;
	if (blurSize % 2 == 0)
		blurSize++;
	GaussianBlur(erod, blur, Size(blurSize, blurSize), 0, 0);
	//namedWindow("GaussianBlur", WINDOW_NORMAL);
	//imshow("GaussianBlur", blur);
	imwrite("D:\\test\\blur.jpg", blur);
	//封装的二值化
	Mat thresh = gray.clone();
	adaptiveThreshold(~gray, thresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, -2);
	//imshow("thresh",thresh);
	imwrite("D:\\test\\thresh.jpg", thresh);
	/*
	这部分的思想是将线条从横纵的方向处理后抽取出来，再进行交叉，矩形的点，进而找到矩形区域的过程
	*/
	// Create the images that will use to extract the horizonta and vertical lines
	Mat horizontal = thresh.clone();
	Mat vertical = thresh.clone();

	int scale = 20; // 这个值越大，检测到的直线越多

	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols*0.05;// / scale;

	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));//449ms

	int verticalsize = vertical.rows*0.165;// / scale;//463  //panlin

	// Create structure element for extracting vertical lines through morphology operations
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));

	// Apply morphology operations
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));//200



	dilate(horizontal, horizontal, cv::Mat());
	dilate(vertical, vertical, cv::Mat());

	//horizontal = a(horizontal);
	//bitwise_not(horizontal, horizontal);
	//bitwise_not(vertical, vertical);
	//vertical = a(vertical);//600
	//Mat first;

	imwrite("D:\\test\\vertical.jpg", vertical);

	Mat first = gray.clone();
	//Mat first = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
	yy(vertical, gray, first);


	//first.release();
	//imwrite("D:/result/first.jpg", first);
	yy2(horizontal, first, first);//
	//imwrite("D:/result/first2.jpg", first);
	////////////////

	resize(first, first, Size(or_src_height, or_src_width), 0, 0, INTER_LINEAR);
	imwrite("D:\\test\\上采.jpg", first);
	//cvtColor(first, first, CV_GRAY2BGR);
	return first;


}



//自适应阈值的Canny，获取low，high两个参数。
void AdaptiveFindThreshold(const cv::Mat src, double *low, double *high, int aperture_size)
{
	const int cn = src.channels();
	cv::Mat dx(src.rows, src.cols, CV_16SC(cn));
	cv::Mat dy(src.rows, src.cols, CV_16SC(cn));

	cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0);
	cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0);

	CvMat _dx = dx, _dy = dy;
	_AdaptiveFindThreshold(&_dx, &_dy, low, high);

}

// 仿照matlab，自适应求高低两个门限                                              
void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high)
{
	CvSize size;
	IplImage *imge = 0;
	int i, j;
	CvHistogram *hist;
	int hist_size = 255;
	float range_0[] = { 0, 256 };
	float* ranges[] = { range_0 };
	double PercentOfPixelsNotEdges = 0.7;
	size = cvGetSize(dx);
	imge = cvCreateImage(size, IPL_DEPTH_32F, 1);
	// 计算边缘的强度, 并存于图像中                                          
	float maxv = 0;
	for (i = 0; i < size.height; i++)
	{
		const short* _dx = (short*)(dx->data.ptr + dx->step*i);
		const short* _dy = (short*)(dy->data.ptr + dy->step*i);
		float* _image = (float *)(imge->imageData + imge->widthStep*i);
		for (j = 0; j < size.width; j++)
		{
			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
			maxv = maxv < _image[j] ? _image[j] : maxv;

		}
	}
	if (maxv == 0) {
		*high = 0;
		*low = 0;
		cvReleaseImage(&imge);
		return;
	}

	// 计算直方图                                                            
	range_0[1] = maxv;
	hist_size = (int)(hist_size > maxv ? maxv : hist_size);
	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(&imge, hist, 0, NULL);
	int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);
	float sum = 0;
	int icount = hist->mat.dim[0].size;

	float *h = (float*)cvPtr1D(hist->bins, 0);
	for (i = 0; i < icount; i++)
	{
		sum += h[i];
		if (sum > total)
			break;
	}
	// 计算高低门限                                                          
	*high = (i + 1) * maxv / hist_size;
	*low = *high * 0.4;
	cvReleaseImage(&imge);
	cvReleaseHist(&hist);
}




Mat yy(Mat src1, Mat src2, Mat dst)
{
	for (int i = 0; i< src1.rows; i++)
	{
		for (int j = 10; j< src1.cols; j++)
		{
			if (src1.at<uchar>(i, j) == 0)
			{
				dst.at<uchar>(i, j) = src2.at<uchar>(i, j);

			}
			else if ((src1.at<uchar>(i, j) == 255))
			{
				//dst.at<uchar>(i, j) = 255;
				dst.at<uchar>(i, j) = src2.at<uchar>(i, j - 5); //panlin

			}
		}
	}


	return dst;


}


Mat yy2(Mat src1, Mat src2, Mat dst)
{

	for (int i = 10; i< src1.rows; i++)
	{
		for (int j = 0; j< src1.cols; j++)
		{
			if (src1.at<uchar>(i, j) == 0)
			{
				dst.at<uchar>(i, j) = src2.at<uchar>(i, j);

			}
			else if ((src1.at<uchar>(i, j) == 255))
			{
				//dst.at<uchar>(i, j) = 255;
				dst.at<uchar>(i, j) = src2.at<uchar>(i - 5, j);  //panlin

			}
		}
	}


	return dst;


}





///////////////////

int main()
{
	//pb使用
	//opencvpb();

	Mat img = imread("D:\\test\\1.jpg");
	nHeight = img.rows / 6.0;  //panlin,将图像截成6行
	img = extractTable(img);
	//imshow("src", img);   
	//img = imread(img, 0);
	imwrite("D:\\test\\0.jpg", img);
	//src
	//Mat img=imread("D:/0.jpg",0);
	resize(img, img, Size(), 1, 1, INTER_LANCZOS4);
	//resize(img, img, Size(img.cols,img.rows), 0, 0);
	//imwrite("D:/1.jpg", img);
	//statistics CutSingle_Char;
	//vector<Mat> chars_set;
	//vector<vector<int>> Matrix_0;

	clock_t clock_1, clock_2;
	clock_1 = clock();
	vector<Mat> chars_set = CutSingleChar(img);            //显示二值化图像，有进行尺度变换
	clock_2 = clock();
	Mat show2 = img.clone();



	for (int i = 0; i < chars_set.size(); i++)
	{
		/*字符识别*/
		Rect r(tid[i].x, tid[i].y, tid[i].width, tid[i].height);                  //(参数1：左上角的x坐标；参数2：左上角y坐标；参数3：矩形宽；参数4：矩形高)
		rectangle(show2, r, Scalar(0, 0, 0), 1, 8, 0);
	}
	imwrite("D:\\test\\show2.jpg", show2);
	double time_cut = ((double)(clock_2 - clock_1) / CLOCKS_PER_SEC);
	cout << "  time:" << time_cut * 1000 << "ms" << endl;

	waitKey();
	return 0;
}



//void opencvpb()
//{
//	//String weights = "D:/test/hccr-80000.pb";
//	//String prototxt = "D:/test/hccr.pbtxt";
//	String weights = "D:/a.pb";
//	String prototxt = "D:/b.pbtxt";
//	dnn::Net net = cv::dnn::readNetFromTensorflow(prototxt, weights);
//	//dnn::Net net = dnn::experimental_dnn_v5::readNetFromTensorflow(weights);
//
//	Mat img = imread("D:/test/0.jpg", 1);
//
//	Mat inputBlob = dnn::blobFromImage(img, 0.00390625f, Size(256, 256), Scalar(), false);
//
//	net.setInput(inputBlob, "data");//set the network input, "data" is the name of the input layer     
//
//	Mat pred = net.forward("fc2/prob");





