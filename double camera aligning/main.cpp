
#include "./helpers/functions.h"


using namespace std;

bool is_zero(cv::Mat& img) {
    if (img.empty()) {
        cerr<<"The image matrix is empty."<<endl;
        return false;
    }

    if (img.type() != CV_64F && img.type() != CV_64FC3) {
        cerr<<"Unsupported image matrix type."<<endl;
        return false;
    }

    if (img.type() == CV_64F) {
        return cv::sum(cv::abs(img))[0] == 0;
    } else {
        vector<cv::Mat> channels;
        cv::split(img, channels);

        for (auto& channel : channels) {
            if (cv::sum(cv::abs(channel))[0] != 0)
                return false;
        }
        return true;
    }
}


int main() {
    matching* cm = new matching();
    cm->left = cm->read_csv("./docs/left_uwp_map.csv");
    cm->right = cm->read_csv("./docs/right_uwp_map.csv");
    
    cv::imshow("Left With Noise", cm->left);
    cv::waitKey(0);
    cv::destroyWindow("Left With Noise");
    cm->remove_noises(cm->left);
    cv::imshow("Left Without Noise", cm->left);
    cv::waitKey(0);
    cv::destroyWindow("Left Without Noise");

    cv::imshow("Right With Noise", cm->right);
    cv::waitKey(0);
    cv::destroyWindow("Right With Noise");
    cm->remove_noises(cm->right);
    cv::imshow("Right Without Noise", cm->right);
    cv::waitKey(0);
    cv::destroyWindow("Right Without Noise");

    cv::Mat left_and_right = cm->correspondance_matching();
    cv::imshow("Correspondance Matched", left_and_right * 255);
    cv::waitKey(0);
    cv::destroyWindow("Correspondance Matched");
    cv::imwrite("./docs/correspondance.png",left_and_right * 255);

    if (is_zero(left_and_right)) {
        cout<<"Image is a zero matrix."<<endl;
    } else {
        cout<<"Image is not a zero matrix."<<endl;
    }
    return 0;
}