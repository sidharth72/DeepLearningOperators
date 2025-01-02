#include "FlattenLayer.h"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmanipulation.hpp"

FlattenLayer::FlattenLayer() {}

xt::xarray<double> FlattenLayer::forward(const xt::xarray<double>& input_data) {
    // Store input shape
    input_shape = std::vector<size_t>(input_data.shape().begin(), input_data.shape().end());
    
    // Calculate batch size and flattened size
    size_t batch_size = input_shape[0];
    size_t flattened_size = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
        flattened_size *= input_shape[i];
    }

    // Reshape the input data
    std::vector<size_t> new_shape = {batch_size, flattened_size};
    return xt::reshape_view(input_data, new_shape);
}

std::vector<size_t> FlattenLayer::get_output_shape() const {
    if (input_shape.empty()) {
        throw std::runtime_error("Input shape not set. Forward pass must be called first.");
    }

    size_t batch_size = input_shape[0];
    size_t flattened_size = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
        flattened_size *= input_shape[i];
    }

    return {batch_size, flattened_size};
}