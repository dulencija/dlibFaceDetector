#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv/cv_image.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/array.hpp>
#include <boost/date_time.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"


using namespace cv;
using namespace std;
using namespace dlib;

//const float DEG2RAD = 3.141593f / 180;
//const float INCH2METER = 0.0254f;


int main(int argc, char *argv[]) 
{
	//DLIB FACE DETECTOR - WORKING SAMPLE

	frontal_face_detector detector = get_frontal_face_detector();

	VideoCapture stream(0);

	if (!stream.isOpened()) {
		cout << "cannot open file";
	}
	int frameCounter = 0;

	while (true)
	{
		std::cout << frameCounter++ << std::endl;

		Mat frame;
		Mat frameGray;
		stream.read(frame);
		cv::cvtColor(frame,  frameGray, CV_RGB2GRAY);

		std::vector<dlib::rectangle> dets = detector(dlib::cv_image<uchar>(frameGray));

		for (int i = 0; i < dets.size(); i++)
		{
			cv::rectangle(frame, cv::Rect(dets[i].left(), dets[i].top(), dets[i].width(), dets[i].height()), cv::Scalar(255, 255, 0, 0), 2);
		}

		imshow("preview", frame);
		if (waitKey(1) >= 0)
			break;
	}

	return 0;
}