#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;

class matching{
    public:
        cv::Mat left;
        cv::Mat right;
        size_t w;
        size_t h;
        size_t c;

        matching(){}
        matching(cv::Mat l, cv::Mat r, size_t width, size_t height, size_t channels){
            left = l;
            right = r;
            w = width;
            h = height;
            c = channels;
        }
        ~matching(){}

        cv::Mat read_csv(string filename);

        cv::Mat impulse_noise_removal(cv::Mat& image, double kernel_size);
        cv::Mat uniform_noise_removal(cv::Mat& image, int kernel_size);
        cv::Mat create_gaussian_kernel(int size, double sigma);
        cv::Mat gaussian_noise_removal(cv::Mat& image, double kernel_size);

        void remove_noises(cv::Mat& image);

        double check_noise(cv::Mat& original, cv::Mat& filtered, double threshold);

        cv::Mat correspondance_matching();
};