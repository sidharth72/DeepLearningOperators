#include "BatchNormalizationLayer.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnpy.hpp"
#include "xtensor/xview.hpp"
#include <iostream>

BatchNormalizationLayer::BatchNormalizationLayer(
    const std::string& gamma_path,
    const std::string& beta_path,
    const std::string& running_mean_path,
    const std::string& running_var_path,
    double epsilon)
    : epsilon(epsilon), momentum(0.1) {
    
    // Load pretrained parameters
    gamma = xt::load_npy<float>(gamma_path);
    beta = xt::load_npy<float>(beta_path);
    running_mean = xt::load_npy<float>(running_mean_path);
    running_var = xt::load_npy<float>(running_var_path);
    
    // Validate shapes
    if (gamma.shape() != beta.shape() ||
        gamma.shape() != running_mean.shape() ||
        gamma.shape() != running_var.shape()) {
        throw std::runtime_error("Parameter shapes don't match");
    }
    
    num_features = gamma.size();
}


// Test mode constructor
BatchNormalizationLayer::BatchNormalizationLayer(
    size_t num_features,
    double epsilon,
    double momentum)
    : num_features(num_features),
      epsilon(epsilon),
      momentum(momentum) {
    
    // Initialize parameters
    gamma = xt::ones<double>({num_features});
    beta = xt::zeros<double>({num_features});
    running_mean = xt::zeros<double>({num_features});
    running_var = xt::ones<double>({num_features});
}


// Testing the input dimensions is correct or not
void BatchNormalizationLayer::_validate_input(const xt::xarray<double>& input_data) {
    if (input_data.dimension() != 4 && input_data.dimension() != 2) {
        throw std::runtime_error(
            "Input must be 4D [batch, height, width, channels] or 2D [batch, features]");
    }
    
    size_t channels = input_data.dimension() == 4 ? 
                     input_data.shape()[3] : 
                     input_data.shape()[1];
                     
    if (channels != num_features) {
        throw std::runtime_error(
            "Number of features/channels doesn't match layer parameters");
    }
}


/* The return type will be a tuple of tensors which
has the mean and variance of the input data
Eg: ({0.3, 0.2, 0.1}, {0.1, 0.2, 0.3})
*/
std::tuple<xt::xarray<double>, xt::xarray<double>> BatchNormalizationLayer::_compute_mean_var(
    const xt::xarray<double>& input_data
){

    std::vector<size_t> axes;

    if (input_data.dimension() == 4){
        axes = {0, 1, 2};
    } else {
        axes = {0};
    }

    /* Compute the mean and variance over the axes
    for {0, 1, 2}, it computes by taking each corresponding channels
    like RED, GREEN, BLUE, computes the overall mean for all RED channels
    all BLUE channels and all GREEN channels and return an array of 3 values.

    For {0}, it computes the mean for all the batch elements returned as a single value
     */
    auto mean = xt::mean(input_data, axes);
    auto centered = input_data - mean;
    auto var = xt::mean(xt::pow(centered, 2), axes);
    return std::make_tuple(mean, var);
}

xt::xarray<double> BatchNormalizationLayer::_normalize(
    const xt::xarray<double>& input_data,
    const xt::xarray<double>& mean,
    const xt::xarray<double>& var
) {
    /* Normalize the input data

        EQ: (x - mean) / sqrt(variance + epsilon)
    */

    auto normalized = (input_data - mean) / xt::sqrt(var + epsilon);
    
    // Scale and shift
    return gamma * normalized + beta;
}

xt::xarray<double> BatchNormalizationLayer::forward(
    const xt::xarray<double>& input_data,
    bool training
) {
    // Validate input
    _validate_input(input_data);
    
    if(training){
        // Compute the batch mean and variance
        auto [batch_mean, batch_var] = _compute_mean_var(input_data);
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean;
        running_var = (1 - momentum) * running_var + momentum * batch_var;

        return _normalize(input_data, batch_mean, batch_var);
    }else{

        // Use the stored running mean and variance for inference
        return _normalize(input_data, running_mean, running_var);
    }
}

void BatchNormalizationLayer::debug_info() const {
    std::cout << "\n=== BatchNormalizationLayer Debug Information ===\n";
    std::cout << "Number of features: " << num_features << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "Momentum: " << momentum << std::endl;
    
    // Print parameter statistics
    std::cout << "\nGamma stats - "
              << "Min: " << xt::amin(gamma)[0] << " "
              << "Max: " << xt::amax(gamma)[0] << " "
              << "Mean: " << xt::mean(gamma)[0] << std::endl;
              
    std::cout << "Beta stats - "
              << "Min: " << xt::amin(beta)[0] << " "
              << "Max: " << xt::amax(beta)[0] << " "
              << "Mean: " << xt::mean(beta)[0] << std::endl;
              
    std::cout << "Running mean stats - "
              << "Min: " << xt::amin(running_mean)[0] << " "
              << "Max: " << xt::amax(running_mean)[0] << " "
              << "Mean: " << xt::mean(running_mean)[0] << std::endl;
              
    std::cout << "Running variance stats - "
              << "Min: " << xt::amin(running_var)[0] << " "
              << "Max: " << xt::amax(running_var)[0] << " "
              << "Mean: " << xt::mean(running_var)[0] << std::endl;
}