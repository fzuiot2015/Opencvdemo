#include <jni.h>
#include<iostream>
#include <algorithm>
#include <time.h>
#include <android/bitmap.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\dnn.hpp>

#ifndef __EXTRACT_TABLE_HPP__
#define __EXTRACT_TABLE_HPP__
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#endif

#define V_PROJECT 1
#define H_PROJECT 2

using namespace cv;
using namespace std;
//定义三个结构体
struct a
{
   int line;
   int x;
   int y;
   int width;
   int height;
   int NUM;
};
struct a tid[1500];

typedef struct
{
   int begin;
   int end;

}char_range_t;

typedef struct Array
{
   int NUM;   //每行字符的总个数
   vector<vector<int>> matrices;  //矩阵信息
}Array;

//函数声明
void BitmapToMat2(JNIEnv *env, jobject& bitmap, Mat& mat, jboolean needUnPremultiplyAlpha);
void BitmapToMat(JNIEnv *env, jobject& bitmap, Mat& mat);
void MatToBitmap2(JNIEnv *env, Mat& mat, jobject& bitmap, jboolean needPremultiplyAlpha);
void MatToBitmap(JNIEnv *env, Mat& mat, jobject& bitmap);
//Mat a(Mat src);
//Mat y(Mat src);
Mat yy(Mat src1, Mat src2, Mat dst);
Mat yy2(Mat src1, Mat src2, Mat dst);
void AdaptiveFindThreshold(const cv::Mat src, double *low, double *high, int aperture_size =3 );
void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high);
jintArray cutPic( JNIEnv *env, Mat& img );

void draw_projection(vector<jint>& pos, jint mode);
int GetTextProjection(Mat &src, vector<jint>& pos, jint mode);
vector<int> GetPeekRange(vector<jint> &vertical_pos, vector<char_range_t> &peek_range, jfloat min_thresh , jint min_range );
//inline void save_cut(const Mat& img, jint id);
Mat cut_one_line(const Mat& src, jint begin, jint end);
vector<Mat> CutSingleChar(Mat& img);

//变量声明
int total_num = 0;
vector<Mat> lines_set;
vector<Mat> chars_set;
int nHeight;

/*
extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_activity_RectCameraActivity_Test( JNIEnv *env, jobject  jobj, jobject jsrcBitmap)
{
    Mat drc;
    BitmapToMat(env,jsrcBitmap,drc);
    resize(drc, drc, Size(drc.cols *0.4, drc.rows *0.4), 0, 0, INTER_LINEAR);
    resize(drc, drc, Size(drc.cols *2.5, drc.rows *2.5), 0, 0, INTER_LINEAR);
    MatToBitmap(env,drc,jsrcBitmap);
    return jsrcBitmap;
}
*/

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_activity_RectCameraActivity_Bitmap2Grey( JNIEnv *env, jobject  jobj, jobject jsrcBitmap)
{

    Mat drc;
    BitmapToMat(env,jsrcBitmap,drc);
    nHeight = drc.rows / 6.0;
    //cvtColor(drc, drc, CV_BGR2GRAY);
    //cvtColor(drc,drc,CV_GRAY2BGRA);
    //resize(drc, drc, Size(drc.cols *0.4, drc.rows *0.4), 0, 0, INTER_LINEAR);
    Mat canny,gray,sobel, edge,erod, blur;
    double src_height=drc.cols, src_width=drc.rows;
    //先转为灰度图
    cvtColor(drc, gray, COLOR_BGR2GRAY);
    //腐蚀（黑色区域变大）
    int erodeSize = src_height*0.003;// 300;

    if (erodeSize % 2 == 0)
       erodeSize++;
    Mat element = getStructuringElement(MORPH_RECT, Size(erodeSize, erodeSize));
    erode(gray, erod, element);
    //高斯模糊化
    int blurSize = src_height*0.005; // 200;
    if (blurSize % 2 == 0)
       blurSize++;
    GaussianBlur(erod, blur, Size(blurSize, blurSize), 0, 0);
    //封装的二值化
    Mat thresh = gray.clone();
    adaptiveThreshold(~gray, thresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, -2);

    //这部分的思想是将线条从横纵的方向处理后抽取出来，再进行交叉，矩形的点，进而找到矩形区域的过程

    // Create the images that will use to extract the horizonta and vertical lines
    Mat horizontal = thresh.clone();
    Mat vertical = thresh.clone();
    int scale = 20; //  这个值越大，检测到的直线越多
    // Specify size on horizontal axis
    int horizontalsize = horizontal.cols*0.05;// / scale;
    // Create structure element for extracting horizontal lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
    // Apply morphology operations
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));//449ms
    int verticalsize = vertical.rows*0.165;// / scale;//463
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
    // Apply morphology operations
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));//200
    dilate(horizontal, horizontal, cv::Mat());
    dilate(vertical, vertical, cv::Mat());
    Mat first=gray.clone();
    yy(vertical, gray, first);
    yy2(horizontal, first, first);

    resize(first, first, Size(src_height , src_width ), 0, 0, INTER_LINEAR);
    MatToBitmap(env,first,jsrcBitmap);


    //新建一个jintArray对象
   // jintArray jntarray = env->NewIntArray(total_num * 5);

     jintArray jntarray = cutPic(env, first);//切割

     return jntarray;



}

void BitmapToMat(JNIEnv *env, jobject& bitmap, Mat& mat)
{
    BitmapToMat2(env, bitmap, mat, false);
}

void BitmapToMat2(JNIEnv *env, jobject& bitmap, Mat& mat, jboolean needUnPremultiplyAlpha)
{
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &dst = mat;
    try {
        //LOGD("nBitmapToMat");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC4);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            //LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            //LOGD("nBitmapToMat: RGB_565 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        //LOGE("nBitmapToMat catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        //LOGE("nBitmapToMat catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}


void MatToBitmap(JNIEnv *env, Mat& mat, jobject& bitmap) {
    MatToBitmap2(env, mat, bitmap, false);
}

void MatToBitmap2(JNIEnv *env, Mat& mat, jobject& bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &src = mat;
    try {
        //LOGD("nMatToBitmap");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        //CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
        //         info.width == (uint32_t) src.cols);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                //LOGD("nMatToBitmap: CV_8UC1 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                //LOGD("nMatToBitmap: CV_8UC3 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if (src.type() == CV_8UC4) {
                //LOGD("nMatToBitmap: CV_8UC4 -> RGBA_8888");
                if (needPremultiplyAlpha)
                    cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else
                    src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                //LOGD("nMatToBitmap: CV_8UC1 -> RGB_565");
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                //LOGD("nMatToBitmap: CV_8UC3 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                //LOGD("nMatToBitmap: CV_8UC4 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        //LOGE("nMatToBitmap catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        //LOGE("nMatToBitmap catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
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
   float range_0[] = { 0,256 };
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


/*
Mat a(Mat src)
{

      //Mat dst;
      int height;
      int width;

      int i;
      int j;

      height = src.rows;
      width = src.cols* src.channels();   // 列项要乘通道数

      //图像反转
      for (i = 0; i< height; i++)
      {
         for (j = 0; j< width; j++)
         {
            src.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);   // 每一个像素反转
         }
      }


      return src;


}

Mat y(Mat src)
{

   //Mat dst;
   int height;
   int width;

   int i;
   int j;

   height = src.rows;
   width = src.cols* src.channels();   // 列项要乘通道数

                              //图像反转
   for (i = 0; i< height; i++)
   {
      for (j = 0; j< width; j++)
      {
         if (src.at<uchar>(i, j) == 255)
         {
            src.at<uchar>(i, j) = 1;
         }
         else
         {
            src.at<uchar>(i, j) = 0;
         }
           // 每一个像素反转
      }
   }


   return src;
}
*/


Mat yy(Mat src1,Mat src2,Mat dst)
{
   //src1掩模
   ///Mat dst;
   //int height;
   //int width;

   //int i;
   //int j;

   //height = src1.rows;
   //width = src1.cols* src1.channels();   // 列项要乘通道数


   for (int i = 0; i< src1.rows; i++)
   {
      for (int j = 10; j< src1.cols; j++)
      {
         if (src1.at<uchar>(i, j) == 0)
         {
            dst.at<uchar>(i, j) = src2.at<uchar>(i, j);

         }
         else if(src1.at<uchar>(i, j) == 255)
         {
            //dst.at<uchar>(i, j) = 255;
            dst.at<uchar>(i, j) = src2.at<uchar>(i, j-5);
            //int t = src2.at<uchar>(i, j - 1);
            //int t2= src2.at<uchar>(i, j );
            //int a=1;
         }
      }
   }


   return dst;


}


Mat yy2(Mat src1, Mat src2, Mat dst)
{
   //src1掩模
   //Mat dst;
   //int height;
   //int width;

   //int i;
   //int j;

   //height = src1.rows;
   //width = src1.cols* src1.channels();   // 列项要乘通道数


   for (int i = 10; i< src1.rows; i++)
   {
      for (int j = 0; j< src1.cols; j++)
      {
         if (src1.at<uchar>(i, j) == 0)
         {
            dst.at<uchar>(i, j) = src2.at<uchar>(i, j);

         }
         else if (src1.at<uchar>(i, j) == 255)
         {
            //dst.at<uchar>(i, j) = 255;
            dst.at<uchar>(i, j) = src2.at<uchar>(i-5, j);

         }
      }
   }
   return dst;
}

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
         uchar* p = src.ptr<uchar>(i);
         for (int j = 0; j < src.cols; j++)
         {
            if (p[j] == 0)
            {
               pos[j]++;
            }
         }
      }

      draw_projection(pos, V_PROJECT);                //水平垂直投影
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
      draw_projection(pos, H_PROJECT);
   }
   return 0;
}

//获取每个分割字符的范围，min_thresh：波峰的最小幅度，min_range：两个波峰的最小间隔        min_range 用于调整两个字符之间的最小间隔 ***
vector<int> GetPeekRange(vector<int> &vertical_pos, vector<char_range_t> &peek_range, float min_thresh , int min_range )
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

//vector<int> compression_params;
//compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//compression_params.push_back(90);//这就是质量 默认值是95
//imwrite("alpha.jpeg",  image_gray, compression_params);

/*
inline void save_cut(const Mat& img, int id)
{
   char name[128] = { 0 };
   sprintf(name, "G:\\characters segmentation\\picture\\test\\%d.jpg", id);
   //imwrite(name, img);
}
*/

//框出字符
Array CutChar(Mat &img, const vector<char_range_t>& v_peek_range, const vector<char_range_t>& h_peek_range, vector<Mat>& chars_set,int line_num)
{
   Array Char;
   vector<vector<int> > Matrix( 50, vector<int>(3) );
   int count = 0;
   Mat show_img = img.clone();      //img.clone()是深拷贝，show_img是img的副本。拷贝的图片和原图完全一样，后面的操作也不会对其产生影响
   cvtColor(show_img, show_img, CV_GRAY2RGB);
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
         //save_cut(single_char, count);

         Matrix[count][0] = x;              //矩阵数据没保留下来？？？？为什么呢
         Matrix[count][1] = width;
         Matrix[count][2] = img.rows;
         count++;
      }
   }

   //imshow("cut", show_img);
   //waitKey();        // 一行行显示代码
   //printf("count = %d", count);

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
   //imshow("binary", img);                                                                       //*******************binary 二值化图
   vector<int> horizion_pos(img.rows, 0);
   vector<char_range_t> h_peek_range;
   GetTextProjection(img, horizion_pos, H_PROJECT);              //获取完整图水平投影
                                                  //获取每个分割字符的范围
   vector<int> y_label = GetPeekRange(horizion_pos, h_peek_range, 8, 40);        //水平分割

#if 1

                                                   /*将每一文本行切割*/
   vector<Mat> lines_set;
   //vector<Mat> lines_set_show;
   for (int i = 0; i < h_peek_range.size(); i++)
   {
      Mat line2 = cut_one_line(img, h_peek_range[i].begin, h_peek_range[i].end); //todo
      lines_set.push_back(line2);
   }

   Array cutchar;
   total_num = 0;

   vector<Mat> chars_set;
   for (int i = 0; i < lines_set.size(); i++)
   {
      Mat line = lines_set[i];
      //Mat line2 = lines_set_show[i];
      //imshow("raw line", line);
      vector<int> vertical_pos(line.cols, 0);
      vector<char_range_t> v_peek_range;
      GetTextProjection(line, vertical_pos, V_PROJECT);           //一行行进行垂直投影
                                                   //获取每个分割字符的范围
      GetPeekRange(vertical_pos, v_peek_range, 0.8, 20);        //垂直分割

      clock_start = clock();
      cutchar = CutChar(line, v_peek_range, h_peek_range, chars_set, i);         //这里进行cut操作    一行行显示出来  ********************************************
      clock_end = clock();

      //typedef struct Array
      //{
      // int NUM;   //每行字符的总个数
      // vector<vector<vector<int> > > matrices;  //矩阵信息
      //}Array;


      int numb_count;

      for (int numb_count = 0; numb_count < cutchar.NUM; numb_count++)
      {
            tid[total_num].x = cutchar.matrices[numb_count][0];

            //if((i*nHeight + (nHeight-line.rows)/2-10)>=0){
            //    tid[total_num].y = i*nHeight + (nHeight-line.rows)/2-10;
            //}else if((i*nHeight + (nHeight-line.rows)/2)>=0){
            //    tid[total_num].y = i*nHeight + (nHeight-line.rows)/2;
            //}else{
            //    tid[total_num].y=0;
            //}

            tid[total_num].y = y_label[i];
            tid[total_num].width = cutchar.matrices[numb_count][1];
            tid[total_num].height = cutchar.matrices[numb_count][2] + 5;//todo
            tid[total_num].line = i;
            total_num = total_num + 1 ;
      }
        //count = total_num;
      //double time = ((double)(clock_end - clock_start) / CLOCKS_PER_SEC);
      //cout << "time:" << time * 1000 << "ms" << endl;
   }
#endif

   return chars_set;
}

jintArray cutPic(JNIEnv *env, Mat& img )
{
      //Mat src, dst;
      //src = imread("sdcard:\\pic.jpg");
      //imwrite("sdcard:\\photo.jpg",src);
      // jint *cbuf;
      // jboolean ptfalse = false;
      //cbuf = env->GetIntArrayElements(buf, &ptfalse);
      //if(cbuf == NULL)
      //{
      // return 0;
      // }
      // Mat img(h, w, CV_8UC4, (unsigned char*)cbuf); // 注意，Android的Bitmap是ARGB四通道,而不是RGB三通道
      //cvtColor(img,img,CV_BGRA2GRAY);
      //cvtColor(img,img,CV_GRAY2BGRA);

      //Mat img = imread("G:\\characters segmentation\\picture\\66.png", 0);
      //imshow("src", img);
    resize(img, img, Size(), 1, 1, INTER_LANCZOS4);
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

    double time_cut = ((double)(clock_2 - clock_1) / CLOCKS_PER_SEC);
    cout << "time:" << time_cut * 1000 << "ms" << endl;

    //waitKey();
/*****
    int size=w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, (jint*)img.data);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;

*****/

    //新建一个jintArray对象
    jintArray jntarray = env->NewIntArray(total_num * 5);
    //获取jntarray对象的指针
    jint * jintp = env->GetIntArrayElements(jntarray, NULL);


    for(jint i = 0; i < total_num * 5; i=i+5){
        if(i == 0){
            jintp[i]   = jint(tid[0].x);
            jintp[i+1] = tid[0].y;
            jintp[i+2] = tid[0].width;
            jintp[i+3] = tid[0].height;
            jintp[i+4] = tid[0].line;
        }
        jintp[i]   = tid[i/5].x;
        jintp[i+1] = tid[i/5].y;
        jintp[i+2] = tid[i/5].width;
        jintp[i+3] = tid[i/5].height;
        jintp[i+4] = tid[i/5].line;
    }

   env->ReleaseIntArrayElements(jntarray, jintp, 0);
    //int size=total_num;
    //jintArray newIntArray = env->NewIntArray(size);
    //env->SetIntArrayRegion(newIntArray, 0, size, intArray);

    return jntarray;

}

