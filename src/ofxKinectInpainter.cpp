#include "ofxKinectInpainter.h"

ofxKinectInpainter::ofxKinectInpainter() {
	scale = 4;
	inpaintRadius = DEFAULT_INPAINT_RADIUS;
}

void ofxKinectInpainter::setup(int width, int height, int downsampling) {
	scale = downsampling;
	this->width = width;
	this->height = height;

#ifndef OFX_CV
	mask.allocate(width, height);
	inpainted.allocate(width, height);
	scaled.allocate((float)width/scale, (float)height/scale);
	scaledMask.allocate((float)width/scale, (float)height/scale);
#else
	mask.create(width, height, CV_8UC1);
	inpainted.create(width, height, CV_8UC1);
	scaled.create((float)width / scale, (float)height / scale, CV_8UC1);
	scaledMask.create((float)width / scale, (float)height / scale, CV_8UC1);
#endif
}


// inpainting happens at a low resolution becuse it's quite
// expensive to compute
#ifndef OFX_CV

void ofxKinectInpainter::inpaint(ofxCvGrayscaleImage &img) {
	// inverted threshold at pix>1 will give a mask of holes only.
	cvThreshold(img.getCvImage(), mask.getCvImage(), 1, 255, CV_THRESH_BINARY_INV);
	mask.flagImageChanged();

	// scale the mask down
	scaledMask.scaleIntoMe(mask);

	// make a scaled version of the image for inpainting
	scaled.scaleIntoMe(img);


	// do the actual inpainting on a low res version
	// using INPAINT_NS is faster than the other option (cv::INPAINT_TELEA seems more stable though)
	cv::Mat imgg = scaled.getCvImage();
	const cv::Mat maskImg = scaledMask.getCvImage();
	cv::inpaint(imgg, maskImg, imgg, inpaintRadius, cv::INPAINT_NS);
	scaled.flagImageChanged();


	// scale up the inpainted image
	inpainted.scaleIntoMe(scaled, CV_INTER_LINEAR);

	//inpainted.convertToRange(0, 100);
	// now blend it with the orignal, using a mask
	cvCopy(inpainted.getCvImage(), img.getCvImage(), mask.getCvImage());
	img.flagImageChanged();
}

#else

void ofxKinectInpainter::inpaint(cv::Mat &img) {
	cv::threshold(img, mask, 1, 255, CV_THRESH_BINARY_INV);

	// scale the mask down
	scaleInto(mask, scaledMask);

	// make a scaled version of the image for inpainting
	scaleInto(img, scaled);

	// do the actual inpainting on a low res version
	// using INPAINT_NS is faster than the other option (cv::INPAINT_TELEA seems more stable though)
	cv::inpaint(scaled, scaledMask, scaled, inpaintRadius, cv::INPAINT_NS);

	// scale up the inpainted image
	scaleInto(scaled, inpainted, cv::INTER_LINEAR);

	//inpainted.convertToRange(0, 100);
	// now blend it with the orignal, using a mask
	inpainted.copyTo(img, mask);
}

#endif

void ofxKinectInpainter::setInpaintingRadius(int radius) {
	inpaintRadius = radius;
}

void ofxKinectInpainter::setDownsampling(int downsampling) {
	if(scale!=downsampling) {
		scale = downsampling;
#ifndef OFX_CV
		scaled.allocate((float)width/scale, (float)height/scale);
		scaledMask.allocate((float)width/scale, (float)height/scale);
#else
		scaled.create((float)width / scale, (float)height / scale, CV_8UC1);
		scaledMask.create((float)width / scale, (float)height / scale, CV_8UC1);
#endif
	}
}

int ofxKinectInpainter::getInpaintingRadius() {
	return inpaintRadius;
}

int ofxKinectInpainter::getDownsampling() {
	return scale;
}

/* private */
void ofxKinectInpainter::scaleInto(cv::Mat &src, cv::Mat &dst, int interp) {
	cv::Size sz = src.size();

	if (sz.width == 0 || sz.height == 0) {
		ofLogError("ofxKinectInpainter") << "scaleInto(): source image is empty";
		return;
	}

	if (dst.channels() != dst.channels() || src.depth() != dst.depth()) {
		ofLogError("ofxKinectInpainter") << "scaleInto(): type mismatch with source image";
		return;
	}

	if (interp != cv::INTER_NEAREST && interp != cv::INTER_LINEAR &&
		interp != cv::INTER_AREA && interp != cv::INTER_CUBIC && interp != cv::INTER_LANCZOS4) {
		ofLogWarning("ofxKinectInpainter") << "scaleInto(): setting interpolationMethod to cv::INTER_NEAREST";
		interp = cv::INTER_NEAREST;
	}

	cv::resize(src, dst, dst.size(), 0, 0, interp);
}
