//
// Created by 刘阳 on 2016/12/18.
//

#define CPU_ONLY

#ifndef CAFFEMODEL_FEATUREEXTRACTION_H
#define CAFFEMODEL_FEATUREEXTRACTION_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

using namespace caffe;

class featureExtraction {

public:
    featureExtraction();
    featureExtraction(std::string model_file, std::string trained_file);
    void load_model(std::string model_file, std::string trained_file);
    void extract_param();
    void extract_feature_map();
    void vis_square(float * feature_blob_data);

    cv::Mat Preprocess(const cv::Mat &img, const std::string& mean_file);
    cv::Mat SetMean(cv::Mat img, const std::string& mean_file);
    void LoadImgs(std::vector<cv::Mat> imgs, const std::string& mean_file);
    void WrapInputLayer(vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels);


    std::shared_ptr<Net<float>> net_;
    std::vector<std::string> blob_names_;
    int num_batch_;
    int num_channel_;
    cv::Size input_geometry_;


};


#endif //CAFFEMODEL_FEATUREEXTRACTION_H
