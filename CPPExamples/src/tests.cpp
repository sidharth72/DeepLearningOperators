#include "layer_tests.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include "DenseLayer.h"
#include "ModelLoader.h"
#include "model_tests.h"


using json = nlohmann::json;
namespace fs = std::filesystem;

// Helper function to get current timestamp as string
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Append test report to JSON file
void append_to_report(const json& report_data, const std::string& test_type = "layer") {
    // Create reports directory if it doesn't exist
    fs::path reports_dir("../report");
    fs::create_directories(reports_dir);

    // Choose report file based on test type
    fs::path report_file = reports_dir /  (test_type == "layer" ? "model_unittest_report.json" : "model_test_report.json");
    std::cout << "Report file: " << report_file << std::endl;
    // Read existing reports
    std::vector<json> existing_reports;
    if (fs::exists(report_file)) {
        try {
            std::ifstream f(report_file);
            json j;
            f >> j;
            if (j.is_array()) {
                existing_reports = j.get<std::vector<json>>();
            } else {
                existing_reports.push_back(j);
            }
        } catch (...) {
            // Start fresh if file is corrupted
        }
    }

    // Append new report
    existing_reports.push_back(report_data);

    // Write atomically using temporary file
    fs::path temp_file = report_file;
    temp_file += ".tmp";
    try {
        std::ofstream f(temp_file);
        f << json(existing_reports).dump(4);
        f.close();
        fs::rename(temp_file, report_file);
    } catch (const std::exception& e) {
        if (fs::exists(temp_file)) {
            fs::remove(temp_file);
        }
        throw;
    }
}

// Execute layer tests and generate report
json run_layer_tests(const std::string& layer_type) {
    auto start_time = std::chrono::system_clock::now();
    TestSuite test_suite;
    json report;

    try {

        std::map<std::string, TestReport> results;

        if (layer_type == "conv") {
            results = test_suite.test_convolution_layer();

        }
        else if (layer_type == "maxpool") {
            results = test_suite.test_max_pooling_layer();
        }
        else if (layer_type == "dense") {
           results = test_suite.test_dense_layer();
        }
        else {
            throw std::runtime_error("Invalid layer type");
        }

        json test_results;
        for (const auto& [test_name, test_report] : results) {
            test_results[test_name] = {
                {"input_shape", test_report.input_shape},
                {"output_shape", test_report.output_shape},
                {"expected_shape", test_report.expected_shape},
                {"min_output", test_report.min_output},
                {"max_output", test_report.max_output},
                {"mean_output", test_report.mean_output}
            };
        }

        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        report = {
            {"timestamp", get_timestamp()},
            {"layer_type", layer_type},
            {"tests_passed", true},
            {"execution_time", std::to_string(elapsed_seconds.count()) + " seconds"},
            {"results", test_results}
        };

        std::cout << "\nTest completed successfully!" << std::endl;
        std::cout << "Layer Type: " << layer_type << std::endl;
        std::cout << "Execution Time: " << elapsed_seconds.count() << " seconds" << std::endl;
        std::cout << "Results saved to reports/model_unittest_report.json" << std::endl;

    } catch (const std::exception& e) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        report = {
            {"timestamp", get_timestamp()},
            {"layer_type", layer_type},
            {"tests_passed", false},
            {"error", e.what()},
            {"execution_time", std::to_string(elapsed_seconds.count()) + " seconds"}
        };

        std::cout << "\nTest failed!" << std::endl;
        std::cout << "Layer Type: " << layer_type << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "Execution Time: " << elapsed_seconds.count() << " seconds" << std::endl;
    }

    return report;
}

int main() {
    try {
        std::string test_type;
        std::cout << "Enter test type: Type 'L' for Layer wise tests, 'M' for Model wise tests: ";
        std::cin >> test_type;

        if (test_type == "L") {
            std::string layer_type;
            std::cout << "Enter layer type (conv/maxpool/dense): ";
            std::cin >> layer_type;

            json report = run_layer_tests(layer_type);
            append_to_report(report);
        }
        else if (test_type == "M") {
            try {
                // Get image path from user
                std::string image_path;
                std::cout << "Enter image path: (path/to/image.jpg) ";
                std::cin >> image_path;

                // Get expected class label
                std::cout << "\nAvailable classes:\n"
                        << "airplane, automobile, bird, cat, deer,\n"
                        << "dog, frog, horse, ship, truck\n\n"
                        << "Enter expected class label: ";
                std::string class_label;
                std::cin >> class_label;

                // Load configuration
                std::string config_path = "../config/network_config.json";
                json config = ModelLoader::load_config(config_path);
                
                // Extract settings
                json global_settings = config["global_settings"];
                json models_config = config["models"];
                std::string base_weights_dir = global_settings["base_weights_directory"];
                std::vector<std::string> class_names = ModelLoader::get_class_names(global_settings);

                // Find expected label index
                auto it = std::find(class_names.begin(), class_names.end(), class_label);
                if (it == class_names.end()) {
                    throw std::runtime_error("Invalid class label: " + class_label);
                }
                size_t expected_label = std::distance(class_names.begin(), it);

                // Preprocess input image
                auto input_data = ModelLoader::preprocess_image(image_path, global_settings);

                // Load model layers (using the second  model with 3x3 filters and 128 dense layer)
                auto layers = ModelLoader::load_model_layers(models_config[1], base_weights_dir);
                

                for (const auto& layer_info : layers) {
                    std::cout << "Layer: " << layer_info.name << std::endl;
                }
                // Convert layers to format expected by test_model_layers
                std::vector<std::pair<std::string, std::shared_ptr<void>>> layer_pairs;
                for (const auto& layer_info : layers) {
                    layer_pairs.emplace_back(layer_info.name, layer_info.layer);
                }

                // Run model tests
                auto results = test_model_layers(layer_pairs, input_data, expected_label, class_names);

                // Create test report
                json model_report = {
                    {"timestamp", get_timestamp()},
                    {"test_type", "model"},
                    {"tests_passed", results.test_passed},
                    {"prediction_correct", results.prediction_correct},
                    {"error", results.error}
                };

                // Save report
                append_to_report(model_report, "model");

                // Print results
                if (results.test_passed) {
                    std::cout << "\nAll test cases passed!" << std::endl;
                    if (results.prediction_correct) {
                        std::cout << "Prediction matches expected label!" << std::endl;
                    } else {
                        std::cout << "Prediction does not match expected label." << std::endl;
                    }
                } else {
                    std::cout << "\nTest failed: " << results.error << std::endl;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error in model tests: " << e.what() << std::endl;
                return 1;
            }
        }
        else {
            std::cout << "Invalid test type" << std::endl;
            return 1;
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}