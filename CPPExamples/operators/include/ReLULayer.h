#pragma once
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xpad.hpp"


class ReLULayer {
    public:
        ReLULayer();
        virtual ~ReLULayer() = default;
        virtual xt::xarray<double> forward(const xt::xarray<double>& input_data);
};

