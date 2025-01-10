#include "ModelLoader.h"
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xtensor/xarray.hpp>
#include <fstream>


json ModelLoader::load_config(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Unable to open config file: " + config_path);
    }
    
    json config;
    config_file >> config;
    return config;
}

std::vector<LayerInfo> ModelLoader::load_model_layers(
    const json& model_config,
    const std::string& base_weights_dir
) {
    std::vector<LayerInfo> layers;
    
    // Get model directory with explicit string conversion
    std::string model_dir = model_config["model_directory"].get<std::string>();
    
    // Iterate through layer configurations
    for (const auto& layer_config : model_config["layers"]) {
        LayerInfo layer_info;
        layer_info.name = layer_config["name"].get<std::string>();
        layer_info.type = layer_config["type"].get<std::string>();
        
        if (layer_info.type == "convolution") {
            // Construct paths with proper string conversions
            std::string filters_path = base_weights_dir + "/" + model_dir + "/" + 
                                     layer_config["weights"]["filters"].get<std::string>();
            std::string biases_path = base_weights_dir + "/" + model_dir + "/" + 
                                    layer_config["weights"]["biases"].get<std::string>();
            
            // Create convolution layer
            auto conv_layer = std::make_shared<ConvolutionLayer>(
                filters_path, 
                biases_path
            );
            layer_info.layer = std::static_pointer_cast<void>(conv_layer);
        }

        else if (layer_info.type == "batch_normalization") {


            std::string gamma = base_weights_dir + "/" + model_dir + "/" + 
                                     layer_config["weights"]["gamma"].get<std::string>();

            std::string beta = base_weights_dir + "/" + model_dir + "/" +
                                    layer_config["weights"]["beta"].get<std::string>();

            std::string running_mean = base_weights_dir + "/" + model_dir + "/" +
                                    layer_config["weights"]["moving_mean"].get<std::string>(); 
            
            std::string running_var = base_weights_dir + "/" + model_dir + "/" +
                                    layer_config["weights"]["moving_variance"].get<std::string>();


            auto batch_norm = std::make_shared<BatchNormalizationLayer>(
                gamma,
                beta,
                running_mean,
                running_var
            );

            layer_info.layer = std::static_pointer_cast<void>(batch_norm);

        }

        else if (layer_info.type == "relu_activation") {
            auto relu_layer = std::make_shared<ReLULayer>();
            layer_info.layer = std::static_pointer_cast<void>(relu_layer);
        }
        else if (layer_info.type == "softmax_activation") {
            auto softmax_layer = std::make_shared<SoftmaxLayer>();
            layer_info.layer = std::static_pointer_cast<void>(softmax_layer);
        }
        else if (layer_info.type == "maxpooling") {
            // Get pool size parameters with explicit integer conversion
            auto pool_size = std::make_tuple(
                layer_config["parameters"]["pool_size"][0].get<int>(),
                layer_config["parameters"]["pool_size"][1].get<int>()
            );
            auto pool_layer = std::make_shared<MaxPoolingLayer>(pool_size);
            layer_info.layer = std::static_pointer_cast<void>(pool_layer);
        }
        else if (layer_info.type == "flatten") {
            auto flatten_layer = std::make_shared<FlattenLayer>();
            layer_info.layer = std::static_pointer_cast<void>(flatten_layer);
        }
        else if (layer_info.type == "dense") {
            // Construct paths with proper string conversions
            std::string weights_path = base_weights_dir + "/" + model_dir + "/" + 
                                     layer_config["weights"]["weights"].get<std::string>();
            std::string biases_path = base_weights_dir + "/" + model_dir + "/" + 
                                    layer_config["weights"]["biases"].get<std::string>();
            
            // Get activation function with default value
            std::string activation;
            if (layer_config["parameters"].contains("activation")) {
                activation = layer_config["parameters"]["activation"].get<std::string>();
            } else {
                activation = "relu";
            }
            
            // Create dense layer
            auto dense_layer = std::make_shared<DenseLayer>(
                weights_path,
                biases_path
            );
            layer_info.layer = std::static_pointer_cast<void>(dense_layer);
        }
        else {
            throw std::runtime_error("Unknown layer type: " + layer_info.type);
        }
        
        layers.push_back(layer_info);
    }
    
    return layers;
}

std::vector<std::string> ModelLoader::get_class_names(const json& global_settings) {
    std::vector<std::string> class_names;
    for (const auto& class_name : global_settings["classes"]) {
        class_names.push_back(class_name);
    }
    return class_names;
}

xt::xarray<float> ModelLoader::preprocess_image(
    const std::string& image_path,
    const json& global_settings
) {
    // Load image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    int height = global_settings["preprocessing"]["input_size"][0];
    int width = global_settings["preprocessing"]["input_size"][1];

    // Convert BGR to RGB first
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // Resize using INTER_AREA for downsampling (similar to PIL's behavior)
    cv::Mat resized;
    if (height < image.rows || width < image.cols) {
        // Use INTER_AREA for downsampling
        cv::resize(rgb_image, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    } else {
        // Use INTER_CUBIC for upsampling
        cv::resize(rgb_image, resized, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
    }

    // Convert to float and normalize
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC1, 1.0f / 255.0f);

    // Create tensor with correct size and shape
    xt::xarray<double> tensor = xt::zeros<float>({1, height, width, 3});

    // Fill the tensor
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv::Vec3f pixel = float_img.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; ++c) {
                tensor(0, h, w, c) = pixel[c];
            }
        }
    }

    std::cout << tensor << std::endl;

    return tensor;
}