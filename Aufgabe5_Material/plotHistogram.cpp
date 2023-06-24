#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

cv::Mat plotHistogram(cv::Mat &image, bool cumulative = false, int histSize = 256);

int main()
{
    cv::Mat img = cv::imread("../schrott.png"); // Read the file
    if(img.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return -1 ;
    }

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);    // In case img is colored

    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Input Image", img);
    cv::Mat hist;
    hist = plotHistogram(img_gray);
    cv::namedWindow("Histogram", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Histogram", hist);
    cv::waitKey(0);
}

cv::Mat plotHistogram(cv::Mat &image, bool cumulative, int histSize){
    // Create Image for Histogram
    int hist_w = 1024; int hist_h = 800;
    int bin_w = cvRound( (double) hist_w/histSize );

    cv::Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 255,255,255) );

    if(image.channels()>1) {
        cerr << "plotHistogram: Please insert only gray images." << endl;
        return histImage;
    }

    // Calculate Histogram
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    cv::Mat hist;
    calcHist( &image, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    if(cumulative) {
        cv::Mat accumulatedHist = hist.clone();
        for (int i = 1; i < histSize; i++) {
            accumulatedHist.at<float>(i) += accumulatedHist.at<float>(i - 1);
        }
        hist = accumulatedHist;
    }

    // Normalize the result to [ 0, histImage.rows ]
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    // Draw bars
    for (int i = 1; i < histSize; i++) {
        cv::rectangle(histImage, Point(bin_w * (i - 1), hist_h),
             Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
             Scalar(50, 50, 50), 1 );
    }

    return histImage;   // Not really call by value, as cv::Mat only saves a pointer to the image data
}
