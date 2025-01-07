#ifndef LAYER_TESTS_H
#define LAYER_TESTS_H

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <iostream>
#include <map>
#include <string>
#include <cassert>


class ConvolutionLayer;
class MaxPoolingLayer;
class DenseLayer;
class ReLULayer;
class SoftmaxLayer;


struct TestReport {

    std::vector <size_t> input_shape;
    std::vector <size_t> output_shape;
    std::vector <size_t> expected_shape;

    double min_output;
    double max_output;
    double mean_output;
};


template <typename T>
double calculate_mean(const T& array);


class TestSuite {
    public:

        static std::map<std::string, TestReport> test_convolution_layer();
        static std::map <std::string, TestReport> test_max_pooling_layer();
        static std::map <std::string, TestReport> test_dense_layer();
        static std::map <std::string, TestReport> test_relu_layer();
        static std::map <std::string, TestReport> test_softmax_layer();

    private:

        struct ConvTestCase {
            std::string name;
            struct {
                size_t num_filters;
                size_t kernel_size;
                size_t input_channels;
            } params;

            std::vector <size_t> input_shape;
            std::string padding;
        };

        struct PoolTestCase {
            std::string name;
            std::pair<size_t, size_t> pool_size;
            std::pair<size_t, size_t> strides;
            std::vector<size_t> input_shape;
        };

        struct DenseTestCase {
            std::string name;
            struct {
                size_t input_size;
                size_t output_size;
                std::string activation;
            } params;
            size_t batch_size;
        };

        struct ReLUTestCase {
            std::string name;
            std::vector<size_t> input_shape;  // Following the pattern from other test cases
        };

        struct SoftmaxTestCase {
            std::string name;
            std::vector<size_t> input_shape;  // {batch_size, height, width, channels}
        };

        static std::vector <ConvTestCase> get_conv_test_cases();
        static std::vector <PoolTestCase> get_pool_test_cases();
        static std::vector <DenseTestCase> get_dense_test_cases();
        static std::vector <ReLUTestCase> get_relu_test_cases();
        static std::vector <SoftmaxTestCase> get_softmax_test_cases();
};

template <typename T>
double calculate_mean(const T& array) {
    return xt::mean(array)();
}

#endif
