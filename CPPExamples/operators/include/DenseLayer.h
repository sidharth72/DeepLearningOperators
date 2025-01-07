#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "xtensor/xarray.hpp"
#include <string>

class DenseLayer {
public:
    // Constructors
    DenseLayer(const std::string& weights_path, const std::string& biases_path);
    
    DenseLayer(size_t input_size, size_t output_size);

    // Forward pass
    xt::xarray<double> forward(const xt::xarray<double>& input_data);

    // Getters
    size_t get_input_size() const { return input_size; }
    size_t get_output_size() const { return output_size; }

private:
    // Layer parameters
    size_t input_size;
    size_t output_size;
    xt::xarray<double> weights;
    xt::xarray<double> biases;

    // Helper functions
    void _validate_input(const xt::xarray<double>& input_data);
};

#endif // DENSE_LAYER_H