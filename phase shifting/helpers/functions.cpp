#pragma once

#include "functions.h"


using namespace std;

sinusoidal::sinusoidal(int width, int height, int patternCount){
    w = width; h = height; channels = 3;
    patterns = new cv::Mat[patternCount];
}

sinusoidal::~sinusoidal() {
    delete[] patterns;
}


cv::Mat* sinusoidal::create_patterns(int pc, double ps, int prd) { // pc = pattern count, ps = phase shift, prd = period

    for (int i = 0; i < pc; ++i) {

        cv::Mat pattern(h, w, CV_64F);

        double phase = i * ps * CV_PI / 180.0;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // Creating pattern with the peroiod and phase
                pattern.at<double>(y, x) = (0.5 * (1 + sin(2 * CV_PI * x / prd + phase)));
            }
        }

        patterns[i] = pattern;

        string fileName = "./patterns/pattern_" + to_string(i + 1) + ".png";
        cv::imwrite(fileName, pattern * 255); // Normalization for 1-0 to 255-0
    }
    return patterns;
}

cv::Mat sinusoidal::create_phase_maps(cv::Mat& I1, cv::Mat& I2, cv::Mat& I3) {
    
    cv::Mat phase_map(h, w, CV_64F);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double i1 = I1.at<double>(y, x);
            double i2 = I2.at<double>(y, x);
            double i3 = I3.at<double>(y, x);
            // Equation shown by the paper
            phase_map.at<double>(y, x) = atan2(sqrt(3) * (i1 - i3), 2 * i2 - i1 - i3);
        }
    }

    return phase_map;
}

cv::Mat sinusoidal::unwrap(cv::Mat& pmp) {
    cv::Mat unwrapped_phase = pmp.clone();
    
    for (int y = 1; y < pmp.rows; ++y) {
        for (int x = 1; x < pmp.cols; ++x) {
            //Calculating phase difference
            double diff_x = pmp.at<double>(y, x) - pmp.at<double>(y, x - 1);
            double diff_y = pmp.at<double>(y, x) - pmp.at<double>(y - 1, x);
            
            // Putting the values between -pi and pi
            if (diff_x > CV_PI) {
                unwrapped_phase.at<double>(y, x) -= 2 * CV_PI;
            } else if (diff_x < -CV_PI) {
                unwrapped_phase.at<double>(y, x) += 2 * CV_PI;
            }
            
            if (diff_y > CV_PI) {
                unwrapped_phase.at<double>(y, x) -= 2 * CV_PI;
            } else if (diff_y < -CV_PI) {
                unwrapped_phase.at<double>(y, x) += 2 * CV_PI;
            }
        }
    }
    
    return unwrapped_phase;
}

cv::Mat sinusoidal::wrap(cv::Mat& pmp) {
    cv::Mat wrapped_phase = pmp.clone();

    for (int y = 0; y < pmp.rows; ++y) {
        for (int x = 0; x < pmp.cols; ++x) {
            double phase = pmp.at<double>(y, x);

            // Putting the values between -pi and pi
            while (phase > CV_PI) {
                phase -= 2 * CV_PI;
            }
            while (phase < -CV_PI) {
                phase += 2 * CV_PI;
            }

            wrapped_phase.at<double>(y, x) = phase;
        }
    }

    return wrapped_phase;
}

cv::Mat sinusoidal::averaged_phase_map(cv::Mat& pmp1, cv::Mat& pmp2) {
    cv::Mat unwrapped_pmp1 = unwrap(pmp1);
    cv::Mat unwrapped_pmp2 = unwrap(pmp2);

    cv::Mat avmp(pmp1.size(), CV_64F);
    
    for (int y = 0; y < pmp1.rows; y++) {
        for (int x = 0; x < pmp1.cols; x++) {
            double phase1 = unwrapped_pmp1.at<double>(y, x);
            double phase2 = unwrapped_pmp2.at<double>(y, x);

            // Averaging 2 waves, averaging sin and cos and then get arctan of them
            double sin_avg = (sin(phase1) + sin(phase2)) / 2.0;
            double cos_avg = (cos(phase1) + cos(phase2)) / 2.0;
            
            avmp.at<double>(y, x) = atan2(sin_avg, cos_avg);
        }
    }

    avmp = wrap(avmp);

    return avmp;
}

int gray_to_binary(int gray) {
    int binary = gray;
    while (gray >>= 1){
        // gray to binay
        binary ^= gray;
    }
    return binary;
}

cv::Mat sinusoidal::decode_gray_images(std::vector<cv::Mat>& gray_images) {
    int rows = gray_images[0].rows;
    int cols = gray_images[0].cols;
    cv::Mat decimal_matrix(rows, cols, CV_8U, cv::Scalar(0));  

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int gray_code = 0;
            for (int k = 0; k < gray_images.size(); k++) {
                // Getting the gray code
                gray_code = (gray_code << 1) | static_cast<int>(gray_images[k].at<uchar>(i, j));
            }
            int binary_code = gray_to_binary(gray_code);
            // Assigning it as an increasing order
            decimal_matrix.at<uchar>(i, j) = static_cast<uchar>((binary_code + (j / 10) ) % 256);
        }
    }
    return decimal_matrix;
}

cv::Mat sinusoidal::unwrap_phase_map_fringe_order(cv::Mat& wrapped_phase_map, cv::Mat& fringe_order_matrix) {
    int rows = wrapped_phase_map.rows;
    int cols = wrapped_phase_map.cols;
    cv::Mat unwrapped_phase_map(rows, cols, CV_64F, cv::Scalar(0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double wrapped_phase = wrapped_phase_map.at<double>(i, j);
            double fringe_order = fringe_order_matrix.at<double>(i, j);

            // Fringe order let us find how many times it has wrapped with the order K
            // Wrapped to unwrap with the fringe order of the gray images
            double unwrapped_phase = wrapped_phase + 2 * CV_PI * fringe_order;
            unwrapped_phase_map.at<double>(i, j) = unwrapped_phase;
        }
    }
    return unwrapped_phase_map;
}