#include "DenseLayer.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xadapt.hpp"
#include <xtensor-blas/xlinalg.hpp> // For linear algebra operations
#include <stdexcept>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>

DenseLayer::DenseLayer(size_t input_size_, size_t output_size_, const std::string& activation_)
    : input_size(input_size_), output_size(output_size_), activation(activation_) {

    // Initialize random weights
    weights = 0.01 * xt::random::randn<double>({input_size, output_size});

    // Initialize biases to zero
    biases = xt::zeros<double>({output_size});
}

// Constructor for loading pretrained weights and biases
DenseLayer::DenseLayer(const std::string& weights_path, const std::string& biases_path, const std::string& activation_)
    : activation(activation_) {

    // Load weights
    weights = xt::load_npy<float>(weights_path);
    //weights = xt::cast<double>(weights_float); // Cast to double if needed

    // Load biases
    biases = xt::load_npy<float>(biases_path);
    //biases = xt::cast<double>(biases_float); // Cast to double if needed

    input_size = weights.shape()[0];
    output_size = weights.shape()[1];
}

void DenseLayer::_validate_input(const xt::xarray<double>& input_data) {
    auto shape = input_data.shape();

    if (shape.size() != 2) {
        throw std::runtime_error(
            "Expected 2D input tensor, got shape with dimension " +
            std::to_string(shape.size()));
    }

    if (shape[1] != input_size) {
        throw std::runtime_error(
            "Input size " + std::to_string(shape[1]) +
            " doesn't match weight matrix input size " +
            std::to_string(input_size));
    }
}

xt::xarray<double> DenseLayer::_relu(const xt::xarray<double>& x) {
    return xt::maximum(x, 0.0);
}

xt::xarray<double> DenseLayer::_softmax(const xt::xarray<double>& x) {
    // Compute the maximum values along axis 1
    auto max_vals = xt::amax(x, {1});
    // Expand dimensions to match the input shape
    auto max_vals_expanded = xt::expand_dims(max_vals, 1);

    // Subtract max values and exponentiate
    auto exp_x = xt::exp(x - max_vals_expanded);

    // Compute the sum of exponentiated values along axis 1
    auto sum_exp = xt::sum(exp_x, {1});
    // Expand dimensions to match the input shape
    auto sum_exp_expanded = xt::expand_dims(sum_exp, 1);

    // Return the normalized result
    return exp_x / sum_exp_expanded;
}

xt::xarray<double> DenseLayer::forward(const xt::xarray<double>& input_data) {
    _validate_input(input_data);

    // Linear transformation
    xt::xarray<double> linear_output = xt::linalg::dot(input_data, weights);
    linear_output = linear_output + biases; // W.x + b

    // Apply activation
    if (activation == "relu") {
        return _relu(linear_output);
    } else if (activation == "softmax") {
        return _softmax(linear_output);
    } else {
        return linear_output;
    }
}
