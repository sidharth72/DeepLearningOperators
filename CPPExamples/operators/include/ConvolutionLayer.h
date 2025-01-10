#pragma once
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xpad.hpp"

class ConvolutionLayer {
    public:

        // Constructor for infernece mode with pretrained weights
        ConvolutionLayer(const std::string& filters_path, const std::string& biases_path);

        // Constructor for test mode with random weights
        ConvolutionLayer(size_t num_filters, size_t kernel_size, size_t input_channels);

        void debug_info() const;

        // Forward pass
        xt::xarray<float> forward(const xt::xarray<float>& input_data, const std::string& padding = "same");

        // helper methods

        std::vector<size_t> get_output_shape(const std::vector<size_t>& input_shape, const std::string& padding="same");

    private:

        xt::xarray<float> _pad_input(const xt::xarray<float>& input_data, const std::string& padding);
        float _convolve_single(const xt::xarray<float>& input_slice, const xt::xarray<float>& kernel);

        xt::xarray<float> filters;
        xt::xarray<float> biases;
        size_t num_filters;
        size_t kernel_size;
        size_t input_channels;

};