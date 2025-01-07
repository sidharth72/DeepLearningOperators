#ifndef BATCH_NORMALIZATION_LAYER_H
#define BATCH_NORMALIZATION_LAYER_H

#include <string>
#include "xtensor/xarray.hpp"

class BatchNormalizationLayer {
public:
    // Constructor for inference mode with pretrained parameters
    BatchNormalizationLayer(const std::string& gamma_path, 
                           const std::string& beta_path,
                           const std::string& running_mean_path,
                           const std::string& running_var_path,
                           double epsilon = 1e-5);
    
    // Constructor for training/testing mode
    BatchNormalizationLayer(size_t num_features, 
                           double epsilon = 1e-5,
                           double momentum = 0.1);
    
    // Forward pass
    xt::xarray<double> forward(const xt::xarray<double>& input_data, 
                              bool training = false);
    
    // Debug information
    void debug_info() const;

private:
    // Layer parameters
    xt::xarray<double> gamma;      // Scale parameter
    xt::xarray<double> beta;       // Shift parameter
    xt::xarray<double> running_mean;
    xt::xarray<double> running_var;
    
    size_t num_features;
    double epsilon;
    double momentum;
    
    // Helper functions
    void _validate_input(const xt::xarray<double>& input_data);
    std::tuple<xt::xarray<double>, xt::xarray<double>> _compute_mean_var(
        const xt::xarray<double>& input_data);
    xt::xarray<double> _normalize(const xt::xarray<double>& input_data,
                                 const xt::xarray<double>& mean,
                                 const xt::xarray<double>& var);
};

#endif // BATCH_NORMALIZATION_LAYER_H