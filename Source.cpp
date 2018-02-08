#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
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
	shape_predictor sp;
	deserialize("D:\\shape_predictor_68_face_landmarks.dat") >> sp;

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

		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector<full_object_detection> shapes;
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
			full_object_detection shape = sp(dlib::cv_image<uchar>(frameGray), dets[j]);
			for (int k = 0; k < shape.num_parts(); k++)
			{
				//cv::rectangle(frame, cv::Rect(shape.part(k).x(), dets[i].top(), dets[i].width(), dets[i].height()), cv::Scalar(255, 255, 0, 0), 2);
				cv::circle(frame, cv::Point(shape.part(k).x(), shape.part(k).y()), 2, cv::Scalar(255, 0, 255));
			}
			shapes.push_back(shape);
		}



		imshow("preview", frame);
		if (waitKey(1) >= 0)
			break;
	}


	////DISTORTION TEST
	//cv::Mat leftImg, rightImg;
	//leftImg = cv::imread("D://imageL_94.jpg");
	//rightImg = cv::imread("D://imageR_94.jpg");

	//cv::imshow("leftImg", leftImg);
	//cv::imshow("rightImg", rightImg);
	////cv::waitKey(0);

	//cv::Mat leftImgDist;
	//cv::Mat rightImgDist;

	//leftImgDist.create(leftImg.size(), leftImg.type());
	//rightImgDist.create(rightImg.size(), rightImg.type());

	//leftImgDist.setTo(0);
	//rightImgDist.setTo(0);

	//float fU;//horizontal component
	//float fV;//vertical comp
	//
	//int cropWidth = leftImg.cols;
	//int cropHeight = leftImg.rows;

	//std::cout << "cropWidth: " << cropWidth << std::endl;
	//std::cout << "cropHeight: " << cropHeight << std::endl;

	//bool eEye = false;//is it right eye?
	////math
	//for (int fVc = 0; fVc < cropHeight; fVc++)
	//{
	//	for (int fUc = 0; fUc < cropWidth; fUc++)
	//	{
	//		fU = fUc / (float)cropWidth;
	//		fV = fVc / (float)cropHeight;

	//		// iPhone parameters
	//		float screen_height_pixel = 750.f;          // Amount of physically pixel. Will be used
	//		float screen_width_pixel = 1334.f;          // to compute the display size in meter.
	//		float edge_screen_distance_meter = 0.004f;  // 0.004 is the boarder from the iPhone
	//		float ppi = 326.0f;                         // Pixel per inch, will be used to compute the screen size in meter

	//		// Zeiss VR one parameters
	//		float screen_lens_distance_meter = 0.037f;
	//		float inter_lens_distance_meter = 0.062f;
	//		float K1 = 0.1;
	//		float K2 = 1.0;
	//		float tray_lenscenter_distance_meter = 0.0; // 0 means, the display is vertically centered to the lens.

	//		// Virtual camera parameters according to CTrackedHMD::GetProjectionRaw()
	//		const float cam_fov_t_deg = 50.0f;  // Top
	//		const float cam_fov_b_deg = 50.0f;  // Bottom
	//		const float cam_fov_l_deg = 50.0f;  // Left
	//		const float cam_fov_r_deg = 50.0f;  // Right (left/right should be the same, otherwise we have to take into account if we render for the left eye or not)

	//		// Below here starts the magic to happen...
	//		float screen_height_meter = screen_height_pixel / ppi * INCH2METER;
	//		float screen_width_meter = screen_width_pixel / ppi * INCH2METER;


	//		// fU and fV are given between 0 and 1. The center of distortion is at 0.5.
	//		// This shifts all values to be between [-0.5, 0.5]
	//		fU = fU - 0.5f;
	//		fV = fV - 0.5f;

	//		// Next we move the origin such that it coincides with the
	//		float offsetX = (inter_lens_distance_meter - screen_width_meter / 2) / screen_width_meter;
	//		float offsetY = 0.f;
	//		if (tray_lenscenter_distance_meter != 0) {
	//			float center_y_meter = tray_lenscenter_distance_meter - edge_screen_distance_meter;
	//			offsetY = (screen_height_meter / 2 - center_y_meter) / screen_height_meter;
	//		}
	//		// And apply it
	//		if (eEye == true) {
	//			fU = fU - offsetX;
	//		}
	//		else {
	//			fU = fU + offsetX;
	//		}
	//		fV = fV + offsetY;

	//		// Further we compute the maximum possible tangent angles in x and y direction
	//		float tanAngle_screen_x = (screen_width_meter / 2) / screen_lens_distance_meter;
	//		float tanAngle_screen_y = screen_height_meter / screen_lens_distance_meter;
	//		// and scale our box (with size 1) to it.
	//		fU = fU * tanAngle_screen_x;
	//		fV = fV * tanAngle_screen_y;
	//		// In other words, u,v are now tangent angles and have a direct meaning to our lens!

	//		// Then we can apply the lens distortion, to compute the angle the
	//		// viewer (looking through the lens) will perceive the "pixel"
	//		float r2 = fU*fU + fV*fV;       // Squared radius
	//		float w = 1 + K1*r2 + K2*r2*r2; // Barrel distortion (https://en.wikipedia.org/wiki/Distortion_(optics))
	//		fU = fU * w;
	//		fV = fV * w;

	//		// (u,v) reprecent now the light ray in the direction that the user will percieve the
	//		// light from the given pixel. So, we need to find out, where we have to look op its
	//		// color in the previously rendered image/texture. We actually need to perform the
	//		// same steps to the image/textrue: It is scaled to a box with lengh one, but in the
	//		// range of [0,1]. Therefore, we
	//		// 1) need to shift the center to of the image/texture coordinate system,
	//		//    such that it coincides with the center of the rendering camera.
	//		// 2) we need to scale the box to the tangent angles to have the same units
	//		//    as our current (u,v) coordinates have.
	//		// To further proceed with (u,v) we do the two steps above inverse and apply them to u,v!
	//		float tanAngle_cam_y = tan(cam_fov_t_deg*DEG2RAD) + tan(cam_fov_b_deg*DEG2RAD);
	//		float tanAngle_cam_x = tan(cam_fov_l_deg*DEG2RAD) + tan(cam_fov_r_deg*DEG2RAD);

	//		// We apply the inverse of 2) to u,v
	//		fU = fU / tanAngle_cam_x;
	//		fV = fV / tanAngle_cam_y;

	//		// And the inverse of 1) (we assume that the camera's field of view is symmetreic):
	//		fU = fU + 0.5f;
	//		fV = fV + 0.5f;
	//		// Finally, u and v are the coordinates in our texture and we can do the color look up

	//		if (!eEye)
	//		{
	//			if ((cropWidth * fU >= 0) && (cropWidth * fU < cropWidth) && (cropHeight * fV > 0) && (cropHeight * fV < cropHeight))
	//			{
	//				
	//				leftImgDist.at<cv::Vec3b>(fUc, fVc) = leftImg.at<cv::Vec3b>(int(cropWidth * fU), int(cropHeight * fV));
	//				////interpolation
	//				//float locationX = cropWidth * fU;
	//				//float locationY = cropHeight * fV;
	//				//int locX0 = std::floor(locationX);
	//				//int locX1 = std::ceil(locationX);
	//				//int locY0 = std::floor(locationY);
	//				//int locY1 = std::ceil(locationY);
	//				//float x = locationX - locX0;
	//				//float y = locationY - locY0;
	//				//cv::Vec3b val0 = leftImg.at<cv::Vec3b>(locX0, locY0) * (1 - x) + leftImg.at<cv::Vec3b>(locX1, locY0) * x;
	//				//cv::Vec3b val1 = leftImg.at<cv::Vec3b>(locX0, locY1) * (1 - x) + leftImg.at<cv::Vec3b>(locX1, locY1) * x;
	//				//	
	//				//leftImgDist.at<cv::Vec3b>(fUc, fVc) = val0 * (1-y) + val1 * y;
	//			}
	//		}
	//		else
	//		{
	//			
	//			if ((cropWidth * fU >= 0) && (cropWidth * fU < cropWidth) && (cropHeight * fV > 0) && (cropHeight * fV < cropHeight))
	//			{

	//				rightImgDist.at<cv::Vec3b>(fUc, fVc) = rightImg.at<cv::Vec3b>(int(cropWidth * fU), int(cropHeight * fV));
	//				//float locationX = cropHeight * fU;
	//				//float locationY = cropWidth * fV;
	//				//int locX0 = std::floor(locationX);
	//				//int locX1 = std::ceil(locationX);
	//				//int locY0 = std::floor(locationY);
	//				//int locY1 = std::ceil(locationY);
	//				//float x = locationX - locX0;
	//				//float y = locationY - locY0;
	//				//cv::Vec3b val0 = rightImg.at<cv::Vec3b>(locX0, locY0) * (1 - x) + rightImg.at<cv::Vec3b>(locX1, locY0) * x;
	//				//cv::Vec3b val1 = rightImg.at<cv::Vec3b>(locX0, locY1) * (1 - x) + rightImg.at<cv::Vec3b>(locX1, locY1) * x;

	//				//rightImgDist.at<cv::Vec3b>(fUc, fVc) = val0 * (1 - y) + val1 * y;
	//			}
	//		}
	//	}
	//}

	//eEye = true;//is it right eye?
	////math 
	//
	//for (int fVc = 0; fVc < cropHeight; fVc++)
	//{
	//	for (int fUc = 0; fUc < cropWidth; fUc++)
	//	{
	//		float fU = fUc / (float)cropWidth;
	//		float fV = fVc / (float)cropHeight;

	//		// iPhone parameters
	//		float screen_height_pixel = 750.f;          // Amount of physically pixel. Will be used
	//		float screen_width_pixel = 1334.f;          // to compute the display size in meter.
	//		float edge_screen_distance_meter = 0.004f;  // 0.004 is the boarder from the iPhone
	//		float ppi = 326.0f;                         // Pixel per inch, will be used to compute the screen size in meter

	//		// Zeiss VR one parameters
	//		float screen_lens_distance_meter = 0.037f;
	//		float inter_lens_distance_meter = 0.062f;
	//		float K1 = 0.1;
	//		float K2 = 1.0;
	//		float tray_lenscenter_distance_meter = 0.0; // 0 means, the display is vertically centered to the lens.

	//		// Virtual camera parameters according to CTrackedHMD::GetProjectionRaw()
	//		const float cam_fov_t_deg = 50.0f;  // Top
	//		const float cam_fov_b_deg = 50.0f;  // Bottom
	//		const float cam_fov_l_deg = 50.0f;  // Left
	//		const float cam_fov_r_deg = 50.0f;  // Right (left/right should be the same, otherwise we have to take into account if we render for the left eye or not)

	//		// Below here starts the magic to happen...
	//		float screen_height_meter = screen_height_pixel / ppi * INCH2METER;
	//		float screen_width_meter = screen_width_pixel / ppi * INCH2METER;


	//		// fU and fV are given between 0 and 1. The center of distortion is at 0.5.
	//		// This shifts all values to be between [-0.5, 0.5]
	//		fU = fU - 0.5f;
	//		fV = fV - 0.5f;

	//		// Next we move the origin such that it coincides with the
	//		float offsetX = (inter_lens_distance_meter - screen_width_meter / 2) / screen_width_meter;
	//		float offsetY = 0.f;
	//		if (tray_lenscenter_distance_meter != 0) {
	//			float center_y_meter = tray_lenscenter_distance_meter - edge_screen_distance_meter;
	//			offsetY = (screen_height_meter / 2 - center_y_meter) / screen_height_meter;
	//		}
	//		// And apply it
	//		if (eEye == true) {
	//			fU = fU - offsetX;
	//		}
	//		else {
	//			fU = fU + offsetX;
	//		}
	//		fV = fV + offsetY;

	//		// Further we compute the maximum possible tangent angles in x and y direction
	//		float tanAngle_screen_x = (screen_width_meter / 2) / screen_lens_distance_meter;
	//		float tanAngle_screen_y = screen_height_meter / screen_lens_distance_meter;
	//		// and scale our box (with size 1) to it.
	//		fU = fU * tanAngle_screen_x;
	//		fV = fV * tanAngle_screen_y;
	//		// In other words, u,v are now tangent angles and have a direct meaning to our lens!

	//		// Then we can apply the lens distortion, to compute the angle the
	//		// viewer (looking through the lens) will perceive the "pixel"
	//		float r2 = fU*fU + fV*fV;       // Squared radius
	//		float w = 1 + K1*r2 + K2*r2*r2; // Barrel distortion (https://en.wikipedia.org/wiki/Distortion_(optics))
	//		fU = fU * w;
	//		fV = fV * w;

	//		// (u,v) reprecent now the light ray in the direction that the user will percieve the
	//		// light from the given pixel. So, we need to find out, where we have to look op its
	//		// color in the previously rendered image/texture. We actually need to perform the
	//		// same steps to the image/textrue: It is scaled to a box with lengh one, but in the
	//		// range of [0,1]. Therefore, we
	//		// 1) need to shift the center to of the image/texture coordinate system,
	//		//    such that it coincides with the center of the rendering camera.
	//		// 2) we need to scale the box to the tangent angles to have the same units
	//		//    as our current (u,v) coordinates have.
	//		// To further proceed with (u,v) we do the two steps above inverse and apply them to u,v!
	//		float tanAngle_cam_y = tan(cam_fov_t_deg*DEG2RAD) + tan(cam_fov_b_deg*DEG2RAD);
	//		float tanAngle_cam_x = tan(cam_fov_l_deg*DEG2RAD) + tan(cam_fov_r_deg*DEG2RAD);

	//		// We apply the inverse of 2) to u,v
	//		fU = fU / tanAngle_cam_x;
	//		fV = fV / tanAngle_cam_y;

	//		// And the inverse of 1) (we assume that the camera's field of view is symmetreic):
	//		fU = fU + 0.5f;
	//		fV = fV + 0.5f;
	//		// Finally, u and v are the coordinates in our texture and we can do the color look up

	//		if (!eEye)
	//		{
	//			if ((cropWidth * fU >= 0) && (cropWidth * fU < cropWidth) && (cropHeight * fV > 0) && (cropHeight * fV < cropHeight))
	//			{

	//				leftImgDist.at<cv::Vec3b>(fUc, fVc) = leftImg.at<cv::Vec3b>(int(cropWidth * fU), int(cropHeight * fV));
	//				////interpolation
	//				//float locationX = cropWidth * fU;
	//				//float locationY = cropHeight * fV;
	//				//int locX0 = std::floor(locationX);
	//				//int locX1 = std::ceil(locationX);
	//				//int locY0 = std::floor(locationY);
	//				//int locY1 = std::ceil(locationY);
	//				//float x = locationX - locX0;
	//				//float y = locationY - locY0;
	//				//cv::Vec3b val0 = leftImg.at<cv::Vec3b>(locX0, locY0) * (1 - x) + leftImg.at<cv::Vec3b>(locX1, locY0) * x;
	//				//cv::Vec3b val1 = leftImg.at<cv::Vec3b>(locX0, locY1) * (1 - x) + leftImg.at<cv::Vec3b>(locX1, locY1) * x;

	//				//leftImgDist.at<cv::Vec3b>(fUc, fVc) = val0 * (1 - y) + val1 * y;
	//			}
	//		}
	//		else
	//		{
	//			if ((cropWidth * fU >= 0) && (cropWidth * fU < cropWidth) && (cropHeight * fV > 0) && (cropHeight * fV < cropHeight))
	//			{
	//				rightImgDist.at<cv::Vec3b>(fUc, fVc) = rightImg.at<cv::Vec3b>(int(cropWidth * fU) , int(cropHeight * fV) );
	//			//	float locationX = cropWidth * fU;
	//			//	float locationY = cropHeight * fV;
	//			//	int locX0 = std::floor(locationX);
	//			//	int locX1 = std::ceil(locationX);
	//			//	int locY0 = std::floor(locationY);
	//			//	int locY1 = std::ceil(locationY);
	//			//	float x = locationX - locX0;
	//			//	float y = locationY - locY0;
	//			//	cv::Vec3b val0 = rightImg.at<cv::Vec3b>(locX0, locY0) * (1 - x) + rightImg.at<cv::Vec3b>(locX1, locY0) * x;
	//			//	cv::Vec3b val1 = rightImg.at<cv::Vec3b>(locX0, locY1) * (1 - x) + rightImg.at<cv::Vec3b>(locX1, locY1) * x;

	//			//	rightImgDist.at<cv::Vec3b>(fUc, fVc) = val0 * (1 - y) + val1 * y;
	//			}
	//		}
	//	}
	//}

	////show
	//cv::imshow("leftImgDist", leftImgDist);
	//cv::imwrite("leftImgDist.jpg", leftImgDist);
	//cv::imshow("rightImgDist", rightImgDist);
	//cv::imwrite("rightImgDist.jpg", rightImgDist);

	////big image
	//cv::Mat bigImg;
	//bigImg.create(leftImg.rows, leftImg.cols * 2, leftImg.type());

	//leftImgDist.copyTo(bigImg(Rect(0, 0, leftImgDist.cols, leftImgDist.rows)));
	//rightImgDist.copyTo(bigImg(Rect(rightImgDist.cols, 0, rightImgDist.cols, rightImgDist.rows)));

	//cv::imshow("bigImg", bigImg);
	//cv::imwrite("bigImg.jpg", bigImg);
	//cv::waitKey(0);

	//return 0;
}