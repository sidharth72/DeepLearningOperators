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
        xt::xarray<double> forward(const xt::xarray<double>& input_data, const std::string& padding = "same");

        // helper methods

        std::vector<size_t> get_output_shape(const std::vector<size_t>& input_shape, const std::string& padding="same");

    private:

        xt::xarray<double> _pad_input(const xt::xarray<double>& input_data, const std::string& padding);
        double _convolve_single(const xt::xarray<double>& input_slice, const xt::xarray<double>& kernel);

        xt::xarray<double> filters;
        xt::xarray<double> biases;
        size_t num_filters;
        size_t kernel_size;
        size_t input_channels;

};