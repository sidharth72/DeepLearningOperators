#ifndef MAXPOOLING_LAYER_H
#define MAXPOOLING_LAYER_H

#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include <tuple>

class MaxPoolingLayer {
public:
    // Constructor
    MaxPoolingLayer(std::tuple<size_t, size_t> pool_size = std::make_tuple(2, 2), 
                   std::tuple<size_t, size_t> strides = std::make_tuple(0, 0));

    // Forward pass
    xt::xarray<float> forward(const xt::xarray<float>& input_data);

    // Get output shape
    std::vector<size_t> get_output_shape(const std::vector<size_t>& input_shape);

private:
    size_t pool_height;
    size_t pool_width;
    size_t stride_height;
    size_t stride_width;
};

#endif // MAXPOOLING_LAYER_H