#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "xtensor/xarray.hpp"
#include <string>

class DenseLayer {
public:
    // Constructors
    DenseLayer(const std::string& weights_path, const std::string& biases_path,
               const std::string& activation = "relu");
    
    DenseLayer(size_t input_size, size_t output_size, 
               const std::string& activation = "relu");

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
    std::string activation;

    // Helper functions
    void _validate_input(const xt::xarray<double>& input_data);
    xt::xarray<double> _relu(const xt::xarray<double>& x);
    xt::xarray<double> _softmax(const xt::xarray<double>& x);
};

#endif // DENSE_LAYER_H