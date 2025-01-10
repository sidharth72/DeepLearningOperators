// SoftmaxLayer.h
#pragma once
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xpad.hpp"

class SoftmaxLayer {
public:
    SoftmaxLayer();
    
    // Forward pass
    xt::xarray<float> forward(const xt::xarray<float>& input_data);
};