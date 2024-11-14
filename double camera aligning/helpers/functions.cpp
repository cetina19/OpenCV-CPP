#pragma once

#include "functions.h"

//#define for_loop(i, size) for(int i = 0; i < size; i++) // for loop preprocessing may have used for faster runtime

using namespace std;

cv::Mat matching::read_csv(string filename){
    ifstream file(filename);
    vector<vector<double>> matrix;
    string line, value;

    while(getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        matrix.push_back(row);
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    w = cols;
    h = rows;

    cv::Mat img(rows, cols, CV_64F);

    for(int i = 0; i < rows; i++) 
        for(int k = 0; k < cols; k++) 
            img.at<double>(i, k) = matrix[i][k];

    return img;
}

cv::Mat matching::impulse_noise_removal(cv::Mat& image, double kernel_size){
    cv::Mat result = image.clone();
    // borders half the size of the kernel in this filtering
    double border = kernel_size / 2;
    for (double y = border; y < image.rows - border; y++) {
        for (double x = border; x < image.cols - border; x++) {
            vector<double> kernel;
            for (double wy = -border; wy <= border; wy++) {
                for (double wx = -border; wx <= border; wx++) {
                    kernel.push_back(image.at<double>(y + wy, x + wx));
                }
            }
            // swapping the median value with current pixel
            nth_element(kernel.begin(), kernel.begin() + kernel.size() / 2, kernel.end()); 
            result.at<double>(y, x) = kernel[kernel.size() / 2];
        }
    }
    return result;
}

double matching::check_noise(cv::Mat& original, cv::Mat& filtered, double threshold) {
    int noise_count = 0;
    for (int y = 0; y < original.rows; y++) {
        for (int x = 0; x < original.cols; x++) {
            if (std::abs(original.at<double>(y, x) - filtered.at<double>(y, x)) > threshold) {
                noise_count++;
            }
        }
    }
    double percentage = (100 * noise_count) / (original.rows * original.cols);
    return percentage;
}

cv::Mat matching::uniform_noise_removal(cv::Mat& input_image, int kernel_size) {
    if (kernel_size % 2 == 0) {
        kernel_size++;  
    }
    cv::Mat output_image;
    input_image.convertTo(input_image, CV_8U);
    //cv::fastNlMeansDenoising(input_image, output_image, 10, kernel_size/4, kernel_size );
    //cv::bilateralFilter(input_image, output_image, kernel_size, 0, 0);
    cv::blur(input_image, output_image, cv::Size(kernel_size, kernel_size));
    //cv::medianBlur(input_image, output_image, kernel_size);
    output_image.convertTo(output_image, CV_64F);
    return output_image;
}

cv::Mat matching::create_gaussian_kernel(int kernel_size, double sigma) {
    cv::Mat kernel(kernel_size, kernel_size, CV_64F);
    double sum = 0.0;
    double half_size = kernel_size / 2;

    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int x = i - half_size;
            int y = j - half_size;
            // gaussian equation
            double new_value = std::exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * CV_PI * sigma * sigma);
            kernel.at<double>(i, j) = new_value;
            sum += new_value;
        }
    }

    // normalization of kernel
    kernel /= sum;

    /*cv::imshow("Kernel",kernel);
    cv::waitKey(0);
    cv::destroyWindow("Kernel");
    cout<<"Sum = "<<sum<<endl;*/

    return kernel;
}

cv::Mat matching::gaussian_noise_removal(cv::Mat& input_image, double kernel_size) {
    if (int(kernel_size) % 2 == 0) 
        kernel_size++; 

    double sigma = kernel_size/6; 
    int half_size = kernel_size / 2;
    cv::Mat kernel = create_gaussian_kernel(kernel_size, sigma);
    cv::Mat output_image; // = input_image.clone(); //for manual implementation


    /*cv::Mat padded_image;
    cv::copyMakeBorder(input_image, padded_image, half_size, half_size, half_size, half_size, cv::BORDER_REFLECT);

    for (int i = half_size; i < padded_image.rows - half_size; i++) {
        for (int j = half_size; j < padded_image.cols - half_size; j++) {
            double sum = 0.0;
            for (int k = -half_size; k <= half_size; k++) {
                for (int l = -half_size; l <= half_size; l++) {
                    sum += padded_image.at<double>(i + k, j + l) * kernel.at<double>(k + half_size, l + half_size);
                }
            }
            output_image.at<double>(i - half_size, j - half_size) = sum;
        }
    }*/


    /*cv::Mat kernelX = cv::getGaussianKernel(kernel_size, sigma, CV_64F);
    cv::Mat kernelY = cv::getGaussianKernel(kernel_size, sigma, CV_64F);
    cv::Mat buffer_image;
    cv::filter2D(input_image, buffer_image, -1, kernelX);
    cv::filter2D(_Buffer_view, output_image, -1, kernelY.t());*/

    cv::GaussianBlur(input_image, output_image, cv::Size(kernel_size,kernel_size), sigma, sigma);
    
    return output_image;
}

void matching::remove_noises(cv::Mat& image){
    cv::imwrite("./docs/image.png",image * 255);

    cv::Mat first = impulse_noise_removal(image,12);
    
    cv::imwrite("./docs/first.png",first * 255);

    cv::Mat second = uniform_noise_removal(first, 12);

    cv::imwrite("./docs/second.png",second * 255);

    cv::Mat third = gaussian_noise_removal(second, 12);

    cv::imwrite("./docs/third.png",third * 255);
    
    image = third.clone();

}

// Phase Correlation
// Perform DFT
// Calculate Cross Power Spectrum
// Perform Inverse DFT
// Finding Peak Location
// Assign Peak Location as Correspondance

cv::Mat matching::correspondance_matching(){
    int rows = left.rows;
    int cols = left.cols;
    cv::Mat correspondences = cv::Mat::zeros(rows, cols, CV_64F);

    for (int row = 0; row < rows; ++row) {
        cv::Mat row1 = left.row(row);
        cv::Mat row2 = right.row(row);

        // Perform DFT
        cv::Mat dft1, dft2;
        cv::dft(row1, dft1, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(row2, dft2, cv::DFT_COMPLEX_OUTPUT);

        // Calculate Cross power spectrum
        cv::Mat crossPowerSpectrum;
        cv::mulSpectrums(dft1, dft2, crossPowerSpectrum, 0, true);
        cv::normalize(crossPowerSpectrum, crossPowerSpectrum, 0, 1, cv::NORM_MINMAX);
        
        cv::Mat inverseCrossPowerSpectrum;
        cv::idft(crossPowerSpectrum, inverseCrossPowerSpectrum, cv::DFT_REAL_OUTPUT);

        // Finding Peak Location
        cv::Point peakLoc;
        cv::minMaxLoc(inverseCrossPowerSpectrum, nullptr, nullptr, nullptr, &peakLoc);

        // Assign Peak location as Correspondence
        for (int col = 0; col < cols; ++col) {
            int x_l = col;
            int x_r = (col + peakLoc.x) % cols;
            
            //Just Correspondances
            //correspondences.at<double>(row, col) = (col + peakLoc.x) % cols;

            //Calculating Disparity Map also
            correspondences.at<double>(row, col) = static_cast<double>(x_l - x_r);
        }
    }

    return correspondences;
}

//correspondences.at<double>(row, col) = (col + peakLoc.x) % cols;