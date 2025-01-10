#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "xtensor/xarray.hpp"
#include <vector>

class FlattenLayer {
public:
    // Constructor
    FlattenLayer();

    // Forward pass
    xt::xarray<float> forward(const xt::xarray<float>& input_data);

    // Get output shape
    std::vector<size_t> get_output_shape() const;

private:
    std::vector<size_t> input_shape;
};

#endif // FLATTEN_LAYER_H