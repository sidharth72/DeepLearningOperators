#include "ReLULayer.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <iostream>
#include <cmath>


// Implementation of ReLU activation layer with the forward pass
ReLULayer::ReLULayer() {}


xt::xarray<float> ReLULayer::forward(const xt::xarray<float>& input_data) {
    return xt::maximum(input_data, 0.0);
}


