#include "MaxPoolingLayer.h"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xreducer.hpp"

// Constructor that accepts the pool size and strides
MaxPoolingLayer::MaxPoolingLayer(std::tuple<size_t, size_t> pool_size, 
                               std::tuple<size_t, size_t> strides) {
    pool_height = std::get<0>(pool_size);
    pool_width = std::get<1>(pool_size);
    
    // If strides is (0,0), use pool_size as strides
    if (std::get<0>(strides) == 0 || std::get<1>(strides) == 0) {
        stride_height = pool_height;
        stride_width = pool_width;
    } else {
        stride_height = std::get<0>(strides);
        stride_width = std::get<1>(strides);
    }
}

// Forward pass implementation
xt::xarray<float> MaxPoolingLayer::forward(const xt::xarray<float>& input_data) {
    auto input_shape = input_data.shape();
    size_t batch_size = input_shape[0];
    size_t input_height = input_shape[1];
    size_t input_width = input_shape[2];
    size_t channels = input_shape[3];

    size_t output_height = (input_height - pool_height) / stride_height + 1;
    size_t output_width = (input_width - pool_width) / stride_width + 1;

    // Initialize output tensor with zeros
    std::vector<size_t> output_shape = {batch_size, output_height, output_width, channels};
    xt::xarray<float> output = xt::zeros<float>(output_shape);

    // Perform max pooling
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < output_height; ++h) {
            size_t h_start = h * stride_height;
            size_t h_end = h_start + pool_height;

            for (size_t w = 0; w < output_width; ++w) {
                size_t w_start = w * stride_width;
                size_t w_end = w_start + pool_width;

                for (size_t c = 0; c < channels; ++c) {
                    // Get the window view and compute maximum
                    auto window = xt::view(input_data, b,
                                         xt::range(h_start, h_end),
                                         xt::range(w_start, w_end),
                                         c);
                    
                    output(b, h, w, c) = xt::amax(window)();
                }
            }
        }
    }

    return output;
}

std::vector<size_t> MaxPoolingLayer::get_output_shape(const std::vector<size_t>& input_shape) {
    size_t batch_size = input_shape[0];
    size_t input_height = input_shape[1];
    size_t input_width = input_shape[2];
    size_t channels = input_shape[3];

    size_t output_height = (input_height - pool_height) / stride_height + 1;
    size_t output_width = (input_width - pool_width) / stride_width + 1;

    return {batch_size, output_height, output_width, channels};
}