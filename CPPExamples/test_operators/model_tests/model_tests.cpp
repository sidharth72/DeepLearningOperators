#include "model_tests.h"
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include "FlattenLayer.h"
#include "DenseLayer.h"
#include "SoftmaxLayer.h"
#include "ReLULayer.h"
#include "BatchNormalizationLayer.h"
#include <iostream>
#include <iomanip>

ModelTestResult test_model_layers(
    const std::vector<std::pair<std::string, std::shared_ptr<void>>>& layers,
    const xt::xarray<float>& input_data,
    size_t expected_label,
    const std::vector<std::string>& class_labels
) {
    ModelTestResult result;
    result.test_passed = true;
    
    std::cout << "\nTesting model layers:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Process through each layer
    auto x = input_data;
    
    try {
        for (const auto& layer_pair : layers) {
            const auto& layer_name = layer_pair.first;
            const auto& layer_ptr = layer_pair.second;
            
            try {
                // Type-specific layer processing
                if (layer_name.find("conv") != std::string::npos) {
                    auto conv_layer = std::static_pointer_cast<ConvolutionLayer>(layer_ptr);
                    x = conv_layer->forward(x, "same");
                }

                else if (layer_name.find("batch_norm") != std::string::npos) {
                    auto batch_norm_layer = std::static_pointer_cast<BatchNormalizationLayer>(layer_ptr);
                    x = batch_norm_layer->forward(x);
                }
                else if (layer_name.find("relu") != std::string::npos) {
                    auto relu_layer = std::static_pointer_cast<ReLULayer>(layer_ptr);
                    x = relu_layer->forward(x);
                }
                else if (layer_name.find("softmax") != std::string::npos) {
                    auto softmax_layer = std::static_pointer_cast<SoftmaxLayer>(layer_ptr);
                    x = softmax_layer->forward(x);
                }
                else if (layer_name.find("pool") != std::string::npos) {
                    auto pool_layer = std::static_pointer_cast<MaxPoolingLayer>(layer_ptr);
                    x = pool_layer->forward(x);
                }
                else if (layer_name.find("flatten") != std::string::npos) {
                    auto flatten_layer = std::static_pointer_cast<FlattenLayer>(layer_ptr);
                    x = flatten_layer->forward(x);
                }
                else if (layer_name.find("dense") != std::string::npos) {
                    auto dense_layer = std::static_pointer_cast<DenseLayer>(layer_ptr);
                    x = dense_layer->forward(x);
                }
                else if (layer_name.find("output") != std::string::npos) {
                    auto output_layer = std::static_pointer_cast<DenseLayer>(layer_ptr);
                    x = output_layer->forward(x);
                }
                
                std::cout << "✓ " << layer_name << " - Passed | Output shape: ";
                for (const auto& dim : x.shape()) {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "✗ " << layer_name << " - Failed: " << e.what() << std::endl;
                result.test_passed = false;
                result.error = "Layer " + layer_name + " failed: " + e.what();
                return result;
            }
        }
        
        // Get prediction
        auto output_data = x.data();
        size_t prediction = std::distance(
            output_data,
            std::max_element(output_data, output_data + class_labels.size())
        );
        
        result.confidence = output_data[prediction];
        result.predicted_class = class_labels[prediction];
        result.expected_class = class_labels[expected_label];
        result.prediction_correct = (prediction == expected_label);
        
        // Print results
        std::cout << "\nTest Results:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "All layers passed: " << std::boolalpha << result.test_passed << std::endl;
        std::cout << "Predicted class: " << result.predicted_class << std::endl;
        std::cout << "Expected class: " << result.expected_class << std::endl;
        std::cout << "Confidence: " << std::fixed << std::setprecision(4) << result.confidence << std::endl;
        
    } catch (const std::exception& e) {
        result.test_passed = false;
        result.error = std::string("Unexpected error: ") + e.what();
    }
    
    return result;
}