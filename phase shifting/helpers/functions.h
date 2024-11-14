#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

class sinusoidal{
    public:
        cv::Mat* patterns = NULL;
        int size;
        int w;
        int h;
        int channels = 3;
        
        sinusoidal(){ w = 1280; h = 720; };
        sinusoidal(int w, int h, int pc);
        ~sinusoidal();

        cv::Mat* create_patterns(int patternCount, double phaseShift, int period);
        cv::Mat create_phase_maps(cv::Mat& I1, cv::Mat& I2, cv::Mat& I3);
        cv::Mat unwrap(cv::Mat& wpmp);
        cv::Mat wrap(cv::Mat& uwpmp);
        
        cv::Mat averaged_phase_map(cv::Mat& pmp1, cv::Mat& pmp2);
        
        cv::Mat decode_gray_images(vector<cv::Mat>& gray_coded_images);
        cv::Mat unwrap_phase_map_fringe_order(cv::Mat& wrapped_phase_map, cv::Mat& fringe_order_matrix);
};