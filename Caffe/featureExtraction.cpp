//
// Created by 刘阳 on 2016/12/18.
//

#include "featureExtraction.h"

featureExtraction::featureExtraction() {}

featureExtraction::featureExtraction(std::string model_file, std::string trained_file)
{
    load_model(model_file, trained_file);
}

void featureExtraction::load_model(std::string model_file, std::string trained_file)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    std::shared_ptr<Net<float>> net;
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(trained_file);

    net_ = net;
    blob_names_ = net->blob_names();

    Blob<float>* input_layer = net->input_blobs()[0];
    num_channel_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}

/*
 * extract paramters based on the index
 * the paramter only exist in
 */
void featureExtraction::extract_param()
{
    int name_index = 0;

    const shared_ptr<Blob<float>> params = net_->params()[name_index];
    int batch_size = params->num();
    int dim_features = params->count() / batch_size;
    int height = params->height();
    int width = params->width();
    int channel = params->channels();

    const float *param_data;

    param_data = params->cpu_data();
    //read data to a vector and find it's max and min value
    vector<float> weight(param_data, param_data + params->count());
    float max_value = *std::max_element(weight.begin(), weight.end());
    float min_value = *std::min_element(weight.begin(), weight.end());

    float a = weight[363];

    //reshape data to each batch and each channel
    vector<vector<float>> weight_reshaped(batch_size);

    int HW = height * width;
    int CHW = channel * HW;
    for(int n = 0; n < batch_size; n++)
    {
        for(int hw = 0; hw < HW; hw++)
        {
//            memcpy(weight_reshaped, &weight + c * HW, HW);
            for(int c = 0; c < channel; c++)
            {
//                weight_reshaped[n].push_back(weight[n * CHW + c * HW + hw]);
                weight_reshaped[n].push_back((weight[n * CHW + c * HW + hw] - min_value) / (max_value - min_value));
            }
        }

    }

    //display
    int CW = channel * width;
    for(int n = 0; n < batch_size; ++n)
    {
        cv::Mat img(height, width, CV_32FC3);

        float* srcData = img.ptr<float>(0);
        std::copy(weight_reshaped[n].begin(), weight_reshaped[n].end(), srcData);

//        for(int j = 0; j < img.rows; j++)
//        {
//            float* srcData = img.ptr<float>(j);
//            std::copy(weight_reshaped[n].begin() + j * CW, weight_reshaped[n].begin() + (j + 1) * CW, srcData);
//            //memcpy(&img_vector[j], &weight_reshaped[n] + j * CW, CW * sizeof(float));
//        }

        cv::Mat img_resized;
        cv::resize(img,img_resized,cv::Size(150,150));

        cv::imshow("test", img_resized);
        cv::waitKey(0);
    }
}



void featureExtraction::extract_feature_map()
{
    int name_index = 1;

    net_->Forward();

    const shared_ptr<Blob<float>> feature_blob = net_->blob_by_name(blob_names_[name_index]);
    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count()/batch_size;
    int height = feature_blob->height();
    int width = feature_blob->width();
    int channel = feature_blob->channels();

    const float * feature_blob_data;
    for(int n = 0; n < batch_size; ++n)
    {
        feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);

        //read data to a vector and find it's max and min value
        vector<float> weight(feature_blob_data, feature_blob_data + dim_features);
        float max_value = *std::max_element(weight.begin(), weight.end());
        float min_value = *std::min_element(weight.begin(), weight.end());

        //reshape data to each channel and
        vector<vector<float>> weight_reshaped(channel);

        int HW = height * width;

//        for(int c = 0; c < channel; c++)
//        {
//            for(int hw = 0; hw < height * width; hw++)
//            {
//                weight_reshaped[c].push_back( (weight[c + hw * channel] - min_value) / (max_value - min_value) );
//            }
//        }

//        for (int c = 0; c < channel; c++)
//        {
//            for (int hw = 0; hw < HW; hw++)
//            {
//                weight_reshaped[c].push_back((weight[c * HW + hw] - min_value) / (max_value - min_value));
//            }
//        }

        int index_temp = 0;
        for(int c = 0; c < channel; c++)
        {
            weight_reshaped[c].push_back((weight[index_temp++] - min_value) / (max_value - min_value));
        }

        for(int d = 0; d < dim_features; d++)
        {
            weight[d] = (weight[d] - min_value) / (max_value - min_value);
        }

        for(int d = 0; d < dim_features; d++)
        {
            for(int c = 0; c < channel; c++)
            {
                for(int hw = 0; hw < channel; hw++)
                {
                    if (weight[d] == weight_reshaped[c][hw])
                        std::cout<<"error";
                }
            }
        }

        //display
        for(int c = 0; c < channel; c++)
        {
            cv::Mat img(height, width, CV_32FC1);

            float* srcData = img.ptr<float>(0);

            std::copy(weight_reshaped[c].begin(), weight_reshaped[c].end(), srcData);
//            std::copy(weight.begin() + c * dim_features, weight.begin() + (c + 1) * dim_features, srcData);

//            vector<float> a;
//            for(int d = 0; d < dim_features; d++)
//            {
//                a.push_back(*(srcData + d));
//                if(a[d] != weight[d])
//                    std::cout<< "different";
//            }

            cv::Mat img_resized;
            cv::resize(img,img_resized,cv::Size(150,150));

            cv::imshow("test", img_resized);
            cv::waitKey(0);
        }



        //display
//        for(int c = 0; c < channel; c++)
//        {
//            cv::Mat img(height, width, CV_32FC1);
//
//            for(int j = 0; j < img.rows; j++)
//            {
//                uchar* srcData = img.ptr<uchar>(j);
//                memcpy(srcData, &weight_reshaped[c] + j * width, width);
//            }
//
//            cv::Mat img_resized;
//            cv::resize(img,img_resized,cv::Size(200,200));
//
//            cv::imshow("test", img_resized);
//            cv::waitKey(0);
//        }

//        string out;
//        datum.SerializeToString(&out);
//        int a = 0;
    }
}

void featureExtraction::vis_square(float * feature_blob_data)
{

}

cv::Mat featureExtraction::Preprocess(const cv::Mat &img, const std::string& mean_file)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channel_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channel_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channel_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channel_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_float;
    if (num_channel_ == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);


    /*
     * mean file is the same dimension with the figure
     */
    cv::Mat sample_normalized;
    cv::Mat img_mean = SetMean(img, mean_file);
    cv::subtract(sample_float, img_mean, sample_normalized);

//    cv::cvtColor(sample_float,sample_float,cv::COLOR_BGR2RGB);
//    sample_float = sample_float.t();

    return sample_float;
}

void featureExtraction::LoadImgs(std::vector<cv::Mat> imgs, const std::string& mean_file)
{
    std::shared_ptr<Net<float>> net = net_;

    num_batch_ = imgs.size();

    for(int i = 0; i < num_batch_; i++)
    {
        imgs[i] = Preprocess(imgs[i], mean_file);
        cv::resize(imgs[i],imgs[i],input_geometry_);
    }

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(num_batch_, num_channel_,
                         input_geometry_.height, input_geometry_.width);
    int num = input_layer->num();
    /* Forward dimension change to all layers. */
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(imgs, &input_channels);

}

void featureExtraction::WrapInputLayer(vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels)
{
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float *input_data = input_layer->mutable_cpu_data();

    for (int j = 0; j < num; j++) {
        //std::vector<cv::Mat> *input_channels;
        for (int k = 0; k < input_layer->channels(); ++k) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
        cv::Mat img = imgs[j];
        cv::split(img, *input_channels);
        input_channels->clear();
    }
}

cv::Mat featureExtraction::SetMean(cv::Mat img, const std::string& mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);


    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);

    if(mean_blob.channels() != num_channel_)
        std::cout << "Number of channels of mean file doesn't match input layer." << std::endl;

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channel_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    cv::Size img_geometry;
    img_geometry = cv::Size(img.cols, img.rows);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean = cv::Mat(img_geometry, mean.type(), channel_mean);

    return mean;

}