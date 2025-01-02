#pragma once

#include <vector>
#include <string>
#include <memory>
#include <xtensor/xarray.hpp>
#include "nlohmann/json.hpp"

struct ModelTestResult {
    bool test_passed;
    bool prediction_correct;
    std::string predicted_class;
    std::string expected_class;
    float confidence;
    std::string error;
};

ModelTestResult test_model_layers(
    const std::vector<std::pair<std::string, std::shared_ptr<void>>>& layers,
    const xt::xarray<float>& input_data,
    size_t expected_label,
    const std::vector<std::string>& class_labels
);


