#include "ConvolutionLayer.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <iostream>
#include <cmath>

ConvolutionLayer::ConvolutionLayer(const std::string& filters_path, const std::string& biases_path) {
    filters = xt::load_npy<float>(filters_path);
    biases = xt::load_npy<float>(biases_path);

    // Ensure filters are in the correct format: [height, width, in_channels, out_channels]
    if (filters.dimension() != 4) {
        throw std::runtime_error("Filters must be 4-dimensional");
    }

    // printing the filters shape
    for (size_t i = 0; i < filters.shape().size(); ++i) {
        std::cout << filters.shape()[i] << " ";
    }
    
    num_filters = filters.shape()[3];
    kernel_size = filters.shape()[0];
    input_channels = filters.shape()[2];
    
    // Validate dimensions
    if (filters.shape()[0] != filters.shape()[1]) {
        throw std::runtime_error("Kernel must be square");
    }
    if (biases.size() != num_filters) {
        throw std::runtime_error("Number of biases must match number of filters");
    }
}

// Test mode constructor
ConvolutionLayer::ConvolutionLayer(size_t num_filters, size_t kernel_size, size_t input_channels) {
    this->num_filters = num_filters;
    this->kernel_size = kernel_size;
    this->input_channels = input_channels;
    
    // Initialize filters with random values
    filters = xt::random::randn<float>({kernel_size, kernel_size, input_channels, num_filters});
    
    // Initialize biases with random values
    biases = xt::random::randn<float>({num_filters});
}

void ConvolutionLayer::debug_info() const {
    std::cout << "\n=== ConvolutionLayer Debug Information ===\n";
    std::cout << "Filter shape: " 
              << filters.shape()[0] << "x" << filters.shape()[1] << "x" 
              << filters.shape()[2] << "x" << filters.shape()[3] << std::endl;
    std::cout << "Number of filters: " << num_filters << std::endl;
    std::cout << "Kernel size: " << kernel_size << std::endl;
    std::cout << "Input channels: " << input_channels << std::endl;
    
    // Print first filter's statistics
    auto first_filter = xt::view(filters, 0, 0, xt::all(), 0);
    std::cout << "First filter stats - "
              << "Min: " << xt::amin(first_filter)[0] << " "
              << "Max: " << xt::amax(first_filter)[0] << " "
              << "Mean: " << xt::mean(first_filter)[0] << std::endl;
    
    // Print first few values of biases
    std::cout << "\nFirst few biases: ";
    for(size_t i = 0; i < std::min(size_t(5), biases.size()); ++i) {
        std::cout << biases(i) << " ";
    }
    std::cout << std::endl;
}


// Padding input (same, valid)
xt::xarray<float> ConvolutionLayer::_pad_input(const xt::xarray<float>& input_data, 
                                               const std::string& padding) {
    if (padding == "valid") {
        return input_data;
    }

    size_t pad_h = (kernel_size - 1) / 2;
    size_t pad_w = (kernel_size - 1) / 2;

    std::vector<std::vector<size_t>> pad_width = {
        {0, 0},        // batch dimension
        {pad_h, pad_h}, // height dimension
        {pad_w, pad_w}, // width dimension
        {0, 0}         // channel dimension
    };

    return xt::pad(input_data, pad_width, xt::pad_mode::constant, 0.0);
}

xt::xarray<float> ConvolutionLayer::forward(const xt::xarray<float>& input_data,
                                           const std::string& padding) {
    // Validate input dimensions
    if (input_data.dimension() != 4) {
        throw std::runtime_error("Input must be 4-dimensional [batch, height, width, channels]");
    }
    if (input_data.shape()[3] != input_channels) {
        throw std::runtime_error("Input channels don't match filter channels");
    }

    size_t batch_size = input_data.shape()[0];
    size_t input_height = input_data.shape()[1];
    size_t input_width = input_data.shape()[2];

    auto padded_input = _pad_input(input_data, padding);

    // Calculate output dimensions
    size_t output_height, output_width;
    if (padding == "same") {
        output_height = input_height;
        output_width = input_width;
    } else {  // valid padding
        output_height = input_height - kernel_size + 1;
        output_width = input_width - kernel_size + 1;
    }

    // Initialize output tensor
    std::vector<size_t> output_shape = {batch_size, output_height, output_width, num_filters};
    xt::xarray<float> output = xt::zeros<float>(output_shape);

    // Perform convolution Operation, 
    // For each batch, each filters, we take each of the single value, do the convolution
    // with the single value, populate them into the appropriate tensors to form the output.

    // Iterating over the the defined output dimensions calculated based on the padding.
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < output_height; ++h) {
            for (size_t w = 0; w < output_width; ++w) {
                for (size_t f = 0; f < num_filters; ++f) {
                    float conv_sum = 0.0;
                    
                    // Iterate over the kernel dimensions
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            for (size_t c = 0; c < input_channels; ++c) {
                                size_t h_index = h + kh; // height of filter slice
                                size_t w_index = w + kw;  // width of the filter slice
                                
                                conv_sum += padded_input(b, h_index, w_index, c) * 
                                          filters(kh, kw, c, f);
                            }
                        }
                    }
                    
                    // Add bias and store result
                    output(b, h, w, f) = conv_sum + biases(f);
                }
            }
        }
    }

    return output;  // ReLU activation
}

std::vector<size_t> ConvolutionLayer::get_output_shape(const std::vector<size_t>& input_shape,
                                                      const std::string& padding) {
    if (input_shape.size() != 4) {
        throw std::runtime_error("Input shape must be 4-dimensional");
    }

    size_t batch_size = input_shape[0];
    size_t input_height = input_shape[1];
    size_t input_width = input_shape[2];

    if (padding == "same") {
        return {batch_size, input_height, input_width, num_filters};
    } else {  // valid padding
        size_t output_height = input_height - kernel_size + 1;
        size_t output_width = input_width - kernel_size + 1;
        return {batch_size, output_height, output_width, num_filters};
    }
}