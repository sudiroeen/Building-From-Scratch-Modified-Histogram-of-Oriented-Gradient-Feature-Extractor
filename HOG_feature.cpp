/*

Copyright:

Sudiro

	[at] SudiroEEN@gmail.com

available on my github:
		
		github.com/sudiroeen

*/


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;
using namespace cv;

#define POSITIVE
// #define NEGATIVE
// #define DEBUG

void takeYAMLdata(vector<Mat>& vmatPN, vector<string>& _vsPN, string rectYAML, string saveHOGyaml, string negPos){
	FileStorage _fs(rectYAML, FileStorage::READ);
	Mat TableMat;
	Mat _index;

	_fs["TabelMat"] >> TableMat;
	_fs["_index"] >> _index;
	_fs.release();

	for(int r=0; r<TableMat.rows; r++){
		if(TableMat.at<int>(r,0) != -1){
			Rect _rect(TableMat.at<int>(r,1), TableMat.at<int>(r,2), TableMat.at<int>(r,3), TableMat.at<int>(r,4));
			stringstream _namaFoto;
			_namaFoto << "../datasetMata/mata_" << r << ".jpg";
			vmatPN.push_back(Mat(imread(_namaFoto.str()), _rect));

			stringstream _namaYAML;
			_namaYAML << saveHOGyaml << negPos << "_mata_" << r << ".yaml";
			_vsPN.push_back(_namaYAML.str());
		}
	}
}

void HOG_feature(Mat img, string _nameToSaveYAML, string negPos){
	img.convertTo(img, CV_32F, 1/255.0);
	imshow("img", img);

	Mat kernelSharp = (Mat_<float>(3,3) << 0., -1., 0., -1., 7., -1., 0., -1., 0.);
	filter2D(img, img, -1, kernelSharp);
	imshow("sharped", img);

	Mat gx, gy; 
	Sobel(img, gx, CV_32F, 1, 0, 1);
	Sobel(img, gy, CV_32F, 0, 1, 1);
	
	Mat agx = abs(gx);
	Mat agy = abs(gx);

	Mat mag, angle; 
	cartToPolar(gx, gy, mag, angle, 1); 

	imshow("magnitude", mag);
	imshow("angle", angle);

	Mat singleMag = Mat(mag.size(), CV_32FC1);
	Mat singleAngle= Mat(angle.size(), CV_32FC1);

	for(int r=0; r<angle.rows; r++){
		for(int c=0; c<angle.cols; c++){
			float pixTheta = 0.0;
			float pixM = 0.0;
			for(int ch = 0; ch<3; ch++){
				if(angle.at<Vec3f>(r,c)[ch] > 180.0){
					angle.at<Vec3f>(r,c)[ch] = 360.0 - angle.at<Vec3f>(r,c)[ch];
				}
				if(pixTheta < angle.at<Vec3f>(r,c)[ch])
					pixTheta = angle.at<Vec3f>(r,c)[ch];
				if(pixM < mag.at<Vec3f>(r,c)[ch])
					pixM = mag.at<Vec3f>(r,c)[ch];
			}

			singleAngle.at<float>(r, c) = pixTheta;
			singleMag.at<float>(r, c) = pixM;
		}
	}

	imshow("singleMag", singleMag);
	imshow("singleAngle", singleAngle/255.0);

	resize(singleMag, singleMag, Size(), 96.0/singleMag.cols, 192.0/singleMag.rows);
	resize(singleAngle, singleAngle, Size(), 96.0/singleAngle.cols, 192.0/singleAngle.rows);

	Mat h8x8 = Mat::zeros(8*16, 9, CV_32FC1);
	
	for(int rr = 0; rr<16; rr++){
		for(int cr = 0; cr<8; cr++){
			Mat roiA = Mat(singleAngle, Rect(cr*12, rr*12, 12, 12) );
			Mat roiM = Mat(singleMag, Rect(cr*12, rr*12, 12, 12) );

			for(int r = 0; r<12; r++){
				for(int c = 0; c<12; c++){
					float nilaiA = roiA.at<float>(r,c);
					float nilaiM = roiM.at<float>(r,c);

					if((nilaiA > 160.0) && (nilaiA <=180.0)){
						h8x8.at<float>(rr*8+cr, 8) += (nilaiA - 160.0)/20.0 * nilaiM;
						h8x8.at<float>(rr*8+cr, 7) += (180.0 - nilaiA)/20.0 * nilaiM;
					}else if((nilaiA >= 0.0) && (nilaiA <= 20)){
						h8x8.at<float>(rr*8+cr, 1) += nilaiA/20.0 * nilaiM;
						h8x8.at<float>(rr*8+cr, 0) += (20.0 - nilaiA)/20.0 * nilaiM;
					}else{						
						for(int s =1; s<8; s++){
							float bb = s*20.0, ba = (s+1)*20.0;
							if((nilaiA > bb) && (nilaiA <= ba)){
								h8x8.at<float>(rr*8+cr, s) += (ba - nilaiA)/20.0 * nilaiM;
								h8x8.at<float>(rr*8+cr, s+1) += (nilaiA - bb)/20.0 * nilaiM;
								break;
							}
						}
					}
				}
			}
		}
	}

	Mat h16x16 = Mat::zeros(105, 36, CV_32FC1);
	for(int rr = 0; rr<105; rr++){
		for(int ukuranB = 0; ukuranB<2; ukuranB++){
			for(int ukuranC = 0; ukuranC<2; ukuranC++){
				for(int sembilan = 0; sembilan<9; sembilan++){
						h16x16.at<float>(rr, (ukuranB*2+ukuranC)*9+ sembilan)
								 = h8x8.at<float>(ukuranB*8+ukuranC, sembilan);
					}
				}
			}
		h16x16.row(rr) /= norm(h16x16.row(rr));
	}

	stringstream ss;
	ss << "feature_" << negPos << "_oc";
	FileStorage _fs_(_nameToSaveYAML, FileStorage::WRITE);
	_fs_ << ss.str() << h16x16;
	_fs_.release();
}


int main(){
	std::pair<vector<Mat>, vector<Mat> > croppedImagePN;
	std::pair<vector<string>, vector<string> > _indexCroppedPN;
	takeYAMLdata(croppedImagePN.first, _indexCroppedPN.first, "pos_oc.yaml", "feature_pos_oc/", "pos");
	takeYAMLdata(croppedImagePN.second, _indexCroppedPN.second, "neg_oc.yaml", "feature_neg_oc/", "neg");

#ifdef POSITIVE
	for(int s=0; s<croppedImagePN.first.size(); s++){
		imshow(_indexCroppedPN.first[s], croppedImagePN.first[s]);
		HOG_feature(croppedImagePN.first[s], _indexCroppedPN.first[s], "pos");
#ifdef DEBUG	
		while(true){
			if(waitKey(10) == 27){
				destroyWindow(_indexCroppedPN.first[s]);
				break;
			}
		}
#endif
	}
#endif


#ifdef NEGATIVE
	for(int s=0; s<croppedImagePN.second.size(); s++){
		imshow(_indexCroppedPN.second[s], croppedImagePN.second[s]);
		HOG_feature(croppedImagePN.second[s], _indexCroppedPN.second[s], "neg");
#ifdef DEBUG	
		while(true){
			if(waitKey(10) == 27){
				destroyWindow(_indexCroppedPN.second[s]);
				break;
			}
		}
#endif
	}
#endif
}