#include <iostream>
#include <vector>
#include <algorithm>
#include "../utilities/ModelLoader.h"

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

    for (const auto& layer_info : layers) {
        if (layer_info.type == "convolution") {
            auto conv_layer = std::static_pointer_cast<ConvolutionLayer>(layer_info.layer);
            x = conv_layer->forward(x, "same");
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
        
        std::cout << model_config["model_name"].get<std::string>() << " - " 
                 << layer_info.name << " output shape: ";
        for (auto dim : x.shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

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

Prediction ensemble_predictions(const std::vector<Prediction>& predictions, 
                             const std::vector<std::string>& class_names) {
    if (predictions.empty()) {
        throw std::runtime_error("No predictions to ensemble");
    }

    // Get the shape of output from first prediction
    auto output_shape = predictions[0].raw_output.shape();
    // Create output array with same shape
    xt::xarray<float> avg_output = xt::zeros<float>(output_shape);
    
    // Sum all outputs manually
    for (const auto& pred : predictions) {
        // Add each element
        for (size_t i = 0; i < class_names.size(); ++i) {
            avg_output(0, i) += pred.raw_output(0, i);
        }
    }
    
    // Divide by number of predictions to get average
    float num_predictions = static_cast<float>(predictions.size());
    for (size_t i = 0; i < class_names.size(); ++i) {
        avg_output(0, i) /= num_predictions;
    }
    
    // Find maximum class manually
    size_t ensemble_prediction = 0;
    float max_val = avg_output(0, 0);
    
    for (size_t i = 0; i < class_names.size(); ++i) {
        if (avg_output(0, i) > max_val) {
            max_val = avg_output(0, i);
            ensemble_prediction = i;
        }
    }
    
    
    return Prediction{
        "ensemble",
        class_names[ensemble_prediction],
        max_val,
        avg_output
    };
}

int main() {
    try {
        // Load configuration
        std::string config_path = "../config/network_config.json";
        json config = ModelLoader::load_config(config_path);
        
        // Extract settings
        json global_settings = config["global_settings"];
        json models_config = config["models"];
        std::string base_weights_dir = global_settings["base_weights_directory"];
        std::vector<std::string> class_names = ModelLoader::get_class_names(global_settings);
        
        // Load and preprocess image
        std::string image_path = "../data/inputs/deer.jpg";
        auto input_data = ModelLoader::preprocess_image(image_path, global_settings);
        
        // Process all models
        std::vector<Prediction> predictions;
        for (const auto& model_config : models_config) {
            try {
                auto prediction = predict_single_model(
                    input_data, model_config, base_weights_dir, class_names
                );
                predictions.push_back(prediction);
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing model " 
                         << model_config["model_name"] 
                         << ": " << e.what() << std::endl;
                continue;
            }
        }
        
        // Print individual model predictions
        std::cout << "\nIndividual Model Predictions:\n";
        std::cout << std::string(50, '-') << std::endl;
        for (const auto& pred : predictions) {
            std::cout << "Model: " << pred.model_name << std::endl;
            std::cout << "Predicted class: " << pred.predicted_class << std::endl;
            std::cout << "Confidence: " << pred.confidence * 100 << "%" << std::endl;
            std::cout << std::string(50, '-') << std::endl;
        }
        
        // Get and print ensemble prediction
        auto ensemble_pred = ensemble_predictions(predictions, class_names);
        std::cout << "\nEnsemble Prediction:\n";
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "Predicted class: " << ensemble_pred.predicted_class << std::endl;
        std::cout << "Confidence: " << ensemble_pred.confidence * 100 << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}