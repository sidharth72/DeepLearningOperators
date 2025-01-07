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

DenseLayer::DenseLayer(size_t input_size_, size_t output_size_)
    : input_size(input_size_), output_size(output_size_) {

    // Initialize random weights
    weights = 0.01 * xt::random::randn<double>({input_size, output_size});

    // Initialize biases to zero
    biases = xt::zeros<double>({output_size});
}

// Constructor for loading pretrained weights and biases
DenseLayer::DenseLayer(const std::string& weights_path, const std::string& biases_path){

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


xt::xarray<double> DenseLayer::forward(const xt::xarray<double>& input_data) {
    _validate_input(input_data);

    // Linear transformation
    xt::xarray<double> linear_output = xt::linalg::dot(input_data, weights);
    linear_output = linear_output + biases; // W.x + b

    return linear_output;
}
