/*
 Author:

 	Sudiro 
 	   [at] SudiroEEN@gmail.com

*/


#include "HOG_feature.hpp"


HOG_feature::HOG_feature(Mat _img, string _nameToSaveYAML)
            :img(_img), nameToSaveYAML(_nameToSaveYAML)
           {}


void HOG_feature::saveFeature(){
   stringstream ss;
	ss << "feature";
	FileStorage _fs_(nameToSaveYAML, FileStorage::WRITE);
	_fs_ << ss.str() << HOG_featureOfImage;
	_fs_.release();

	cout << "saved " << nameToSaveYAML << endl;
}


void HOG_feature::getHOG_feature(){
	img.convertTo(img, CV_32F, 1/255.0);

	Mat kernelSharp = (Mat_<float>(3,3) << 0., -1., 0., -1., 7., -1., 0., -1., 0.);
	filter2D(img, img, -1, kernelSharp);
   
	Mat gx, gy; 
	Sobel(img, gx, CV_32F, 1, 0, 1);
	Sobel(img, gy, CV_32F, 0, 1, 1);
	
	Mat agx = abs(gx);
	Mat agy = abs(gx);
	imshow("agx", gx);
	imshow("agy", gy);

	Mat mag, angle; 
	cartToPolar(gx, gy, mag, angle, 1); 

	Mat singleMag = Mat(mag.size(), CV_32FC1);
	Mat singleAngle= Mat(angle.size(), CV_32FC1);

// #ifdef DEBUG
    imshow("img", img);
	imshow("sharped", img);
	imshow("magnitude", mag);
	imshow("angle", angle/255.0);
	imshow("singleMag", singleMag);
	imshow("singleAngle", singleAngle/255.0);
	waitKey(1);
// #endif	

	// string DirOfCropped = "../../datasetGMM_/";

	Mat angp255 = angle/255.0;
	Mat _m, _a;
	mag.convertTo(_m, CV_8U);
	angp255.convertTo(_a, CV_8U);
	imwrite("../../datasetGMM_/mag.jpg", _m);
	
	imwrite("../../datasetGMM_/angp255.jpg", _a);

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

	HOG_featureOfImage = h16x16;
}
