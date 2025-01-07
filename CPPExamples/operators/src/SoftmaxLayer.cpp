#include "SoftmaxLayer.h"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <stdexcept>

SoftmaxLayer::SoftmaxLayer() {}

xt::xarray<double> SoftmaxLayer::forward(const xt::xarray<double>& input_data) {
    if (input_data.dimension() != 2) {
        throw std::invalid_argument("Input must be 2-dimensional (batch_size x features)");
    }

    try {
        // Find max value for numerical stability, keeping axes for proper broadcasting
        auto max_vals = xt::amax(input_data, {1});
        
        // Shift input values by subtracting max (for numerical stability)
        auto shifted = input_data - max_vals;
        
        // Compute exponentials of shifted values
        auto exps = xt::exp(shifted);
        
        // Compute sum of exponentials for normalization
        auto sum_exps = xt::sum(exps, {1});
        
        // Normalize to get probabilities
        auto output = xt::eval(exps / sum_exps);
        
        
        // Validate output
        if (!xt::all(xt::isfinite(output))) {
            throw std::runtime_error("Numerical error: NaN or Inf values detected in output");
        }

        return output;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in softmax forward pass: ") + e.what());
    }
}