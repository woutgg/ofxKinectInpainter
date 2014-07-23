/*
 * Note: to use with ofxCv instead of ofxOpenCv, add '-DOFX_CV' to 'Other C++ Flags' (under LLVM Custom compiler flags)
 */
#pragma once

#ifndef OFX_CV
# include "ofxCvGrayscaleImage.h"
#else
# include "ofxCv.h"
#endif

class ofxKinectInpainter {
public:
	ofxKinectInpainter();
	/**
	 * The amount of downsampling will determine the quality.
	 * 4 is quarter-sized, but real-time.
	 */
	void setup(int width = 640, int height = 480, int downsampling = 4);
	
	/**
	 * This is a parameter of the inpainting algorithm, the higher the better
	 * but the higher the most computationally expensive.
	 */
	void setInpaintingRadius(int radius);
	void setDownsampling(int downsampling);
	
	int getInpaintingRadius();
	int getDownsampling();

#ifndef OFX_CV
	void inpaint(ofxCvGrayscaleImage &img);
#else
	void inpaint(cv::Mat &img);
#endif

private:
#ifndef OFX_CV
	ofxCvGrayscaleImage scaled;
	ofxCvGrayscaleImage mask;
	ofxCvGrayscaleImage scaledMask;
	
	ofxCvGrayscaleImage inpainted;
#else
	cv::Mat scaled;
	cv::Mat mask;
	cv::Mat scaledMask;

	cv::Mat inpainted;
#endif

	float scale;
	int width;
	int height;
	static const int DEFAULT_INPAINT_RADIUS = 3;
	int inpaintRadius;

	void scaleInto(cv::Mat &src, cv::Mat &dst, int interpolation = cv::INTER_LINEAR);
};
