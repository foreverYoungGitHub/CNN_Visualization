#include <iostream>
#include "featureExtraction.h"
using namespace std;

int main() {
    vector<string> model_file = {
            "./model/caffenet.prototxt",
            "./model/det1.prototxt",
            "./model/det2.prototxt",
            "./model/det3.prototxt"
//            "../model/det4.prototxt"
    };

    vector<string> trained_file = {
            "./model/bvlc_reference_caffenet.caffemodel",
            "./model/det1.caffemodel",
            "./model/det2.caffemodel",
            "./model/det3.caffemodel"
//            "../model/det4.caffemodel"
    };

    vector<string> img_file = {
            "./test/cat.jpg"
    };

    vector<cv::Mat> imgs(img_file.size());
    for(int i = 0; i < img_file.size(); i++)
    {
        imgs[i] = cv::imread(img_file[i]);
    }

    string mean_file = "./model/imagenet_mean.binaryproto";

    featureExtraction net(model_file[0], trained_file[0]);
    net.LoadImgs(imgs, mean_file);
    net.extract_feature_map();
//    net.extract_param();

    return 0;
}