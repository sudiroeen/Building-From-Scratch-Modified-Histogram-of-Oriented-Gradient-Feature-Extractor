/*
 Author:

 	Sudiro 
 	   [at] SudiroEEN@gmail.com
*/

#ifndef HOG_FEATURE_HPP
#define HOG_FEATURE_HPP

#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

// #define DEBUG

class HOG_feature{
private:
   string nameToSaveYAML;
   Mat HOG_featureOfImage;
   Mat img;

public:
   HOG_feature(Mat _img, string _nameToSaveYAML);
   void getHOG_feature();
   void saveFeature();
};
#endif
