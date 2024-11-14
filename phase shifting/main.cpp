
#include "./helpers/functions.h"

using namespace std;



int main() {
    sinusoidal* x = new sinusoidal(1280,720,6); 
    cv::Mat* patterns = x->create_patterns(6,60,10);
    
    // For showing the patterns
    /*for(int i=0; i<6; i++){
        string windowName = "Pattern " + to_string(i+1); 
        cv::imshow(windowName, patterns[i]);
        cv::waitKey(0);
        cv::destroyWindow(windowName);
    }*/

   // Double phase has taken as it is written on the paper so phase difference is 120 degrees
    cv::Mat pmp1 = x->create_phase_maps(patterns[0], patterns[2], patterns[4]);
    cv::imshow("Phase Map 1", pmp1 * 255);
    cv::waitKey(0);
    cv::destroyWindow("Phase Map 1");

    cv::imwrite("phase_map_1.png", pmp1 * 255);

    cv::Mat pmp2 = x->create_phase_maps(patterns[1], patterns[3], patterns[5]);
    cv::imshow("Phase Map 2", pmp2 * 255);
    cv::waitKey(0);
    cv::destroyWindow("Phase Map 2");

    cv::imwrite("phase_map_2.png", pmp2 * 255);

    cv::Mat avmp =  x->averaged_phase_map(pmp1,pmp2);
    cv::imshow("Averaged Phase Map", avmp);
    cv::waitKey(0);
    cv::destroyWindow("Averaged Phase Map");

    cv::imwrite("averaged_phase_map.png", avmp * 255);

    vector<cv::Mat> gray_images;
    for(int i=0;i<7;i++){
        cv::Mat img = cv::imread("./imgs/gray_pattern_"+to_string(i)+".png",cv::IMREAD_GRAYSCALE);
        gray_images.push_back(img);
    
    }

    cv::Mat decimal_matrix = x->decode_gray_images(gray_images);
    cv::imshow("Decimal Matrix", decimal_matrix * 255);
    cv::waitKey(0);
    cv::destroyWindow("Decimal Matrix");
    cv::imwrite("decimal_matrix.png", decimal_matrix * 255);
    decimal_matrix.convertTo(decimal_matrix, CV_64F);

    cv::Mat final = x->unwrap_phase_map_fringe_order(avmp,decimal_matrix); 
    cv::imshow("Final", final * 255);
    cv::waitKey(0);
    cv::destroyWindow("Final");
    cv::imwrite("final.png",final * 255);

    cv::imshow("Single Row", final.row(0) * 255);
    cv::waitKey(0);
    cv::destroyWindow("Single Row");
    return 0;
}