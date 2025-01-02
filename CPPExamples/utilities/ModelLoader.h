#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <xtensor/xarray.hpp>
#include <fstream>
#include <memory>
#include "../include/ConvolutionLayer.h"
#include "../include/MaxPoolingLayer.h"
#include "../include/FlattenLayer.h"
#include "../include/DenseLayer.h"

using json = nlohmann::json;

struct LayerInfo {
    std::string name;
    std::shared_ptr<void> layer;
    std::string type;
};

class ModelLoader {
public:
    // Load and parse JSON configuration file
    static json load_config(const std::string& config_path);

    // Load layers for a single model
    static std::vector<LayerInfo> load_model_layers(
        const json& model_config,
        const std::string& base_weights_dir
    );

    // Get class names from global settings
    static std::vector<std::string> get_class_names(const json& global_settings);

    // Load and preprocess image according to global settings
    static xt::xarray<float> preprocess_image(
        const std::string& image_path,
        const json& global_settings
    );
};