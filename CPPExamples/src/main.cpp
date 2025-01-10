#include <iostream>
#include <vector>
#include <algorithm>
#include "../utilities/ModelLoader.h"
#include <chrono>
#include <string>


struct Prediction {
    std::string model_name;
    std::string predicted_class;
    float confidence;
    xt::xarray<float> raw_output;
};

// Single prediction

Prediction predict_single_model(
    const xt::xarray<float>& input_data,
    const json& model_config,
    const std::string& base_weights_dir,
    const std::vector<std::string>& class_names
) {
    std::cout << "\nProcessing model: " << model_config["model_name"] << std::endl;
    
    auto layers = ModelLoader::load_model_layers(model_config, base_weights_dir);
    
    auto x = input_data;
    auto total_duration = 0;

    for (const auto& layer_info : layers) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Layer: " << layer_info.name << std::endl;
        if (layer_info.type == "convolution") {
            
            auto conv_layer = std::static_pointer_cast<ConvolutionLayer>(layer_info.layer);
            x = conv_layer->forward(x, "same");

        }

        else if(layer_info.type == "batch_normalization") {
            auto batch_norm_layer = std::static_pointer_cast<BatchNormalizationLayer>(layer_info.layer);
            x = batch_norm_layer->forward(x);

        }

        else if(layer_info.type == "relu_activation") {
            auto relu_layer = std::static_pointer_cast<ReLULayer>(layer_info.layer);
            x = relu_layer->forward(x);
        }
        else if (layer_info.type == "softmax_activation") {
            auto softmax_layer = std::static_pointer_cast<SoftmaxLayer>(layer_info.layer);
            x = softmax_layer->forward(x);
        }
        else if (layer_info.type == "maxpooling") {
            auto pool_layer = std::static_pointer_cast<MaxPoolingLayer>(layer_info.layer);
            x = pool_layer->forward(x);
        }
        else if (layer_info.type == "flatten") {
            auto flatten_layer = std::static_pointer_cast<FlattenLayer>(layer_info.layer);
            x = flatten_layer->forward(x);
        }
        else if (layer_info.type == "dense") {
            auto dense_layer = std::static_pointer_cast<DenseLayer>(layer_info.layer);
            x = dense_layer->forward(x);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        total_duration = total_duration + duration.count();
        std::cout << "" << model_config["model_name"].get<std::string>() << " - " << layer_info.name << "\n";
        std::cout << "Shape: ";
        auto shape = x.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << " x ";
        }
        std::cout << "\nInference time: " << duration.count() << " ms\n";
        std::cout << "-------------------------------------------------\n";
    }

    std::cout << "\nTotal Inference duration: " << total_duration / 1000.0 << " s" << std::endl;

    // Get prediction
    auto output_data = x.data();
    size_t pred_class = std::distance(
        output_data,
        std::max_element(output_data, output_data + class_names.size())
    );
    
    return Prediction{
        model_config["model_name"],
        class_names[pred_class],
        output_data[pred_class],
        x
    };
}

void print_usage() {
    std::cout << "Usage: " << std::endl;
    std::cout << "Single model: main.exe S <path/to/image>" << std::endl;
    std::cout << "Ensemble    : main.exe E <path/to/image>" << std::endl;
}



// main function that accepts an image path as an argument and processes it
int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage();
        return 1;
    }
    // std::string inference_type = argv[1];
    std::string imagePath = argv[1];

    try {
        // Load configuration
        std::string config_path = "../config/network_config.json";
        std::cout << "Loading configuration from: " << config_path << std::endl;
        json config = ModelLoader::load_config(config_path);



        // Extract settings
        json global_settings = config["global_settings"];
        json models_config = config["models"];

        std::string base_weights_dir = global_settings["base_weights_directory"];
        std::vector<std::string> class_names = ModelLoader::get_class_names(global_settings);
        
        // Load and preprocess image
        auto input_data = ModelLoader::preprocess_image(imagePath, global_settings);

        std::vector<Prediction> predictions;
            // Process single model
        auto model_config = models_config[0];

        auto prediction = predict_single_model(
            input_data, model_config, base_weights_dir, class_names
        );

        predictions.push_back(prediction);
        // Print individual model predictions
        std::cout << "\nModel Predictions:\n";
        std::cout << std::string(50, '-') << std::endl;
        for (const auto& pred : predictions) {
            std::cout << "Model: " << pred.model_name << std::endl;
            std::cout << "Predicted class: " << pred.predicted_class << std::endl;
            std::cout << "Confidence: " << pred.confidence * 100 << "%" << std::endl;
            std::cout << std::string(50, '-') << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}