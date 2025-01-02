#include "layer_tests.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include <iostream>
#include <cassert>
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include "DenseLayer.h"


std::vector<TestSuite::ConvTestCase> TestSuite::get_conv_test_cases() {
    return {
        {
            "Basic test with same padding",
            {32, 3, 3},
            {1, 32, 32, 3},
            "same"
        },
        {
            "Larger input with valid padding",
            {64, 5, 3},
            {4, 64, 64, 3},
            "valid"
        },
        {
            "Single channel input",
            {16, 3, 1},
            {2, 28, 28, 1},
            "same"
        },
        {
            "Large kernel with valid padding",
            {32, 5, 3},
            {1, 64, 64, 3},
            "valid"
        },
        {
            "Large kernel with same padding",
            {64, 7, 3},
            {1, 64, 64, 3},
            "same"
        }
    };
}


std::vector<TestSuite::PoolTestCase> TestSuite::get_pool_test_cases() {
    return {
        {
            "Basic 2x2 pooling with default stride",
            {2, 2},
            {2, 2},
            {1, 4, 4, 3}
        },
        {
            "Custom pooling with different stride",
            {3, 3},
            {2, 2},
            {2, 8, 8, 1}
        },
        {
            "Large input with default pooling",
            {2, 2},
            {2, 2},
            {4, 32, 32, 64}
        },
        {
            "Custom pooling with same stride",
            {3, 3},
            {1, 1},
            {1, 16, 16, 3}
        },
        {
            "Single channel input",
            {2, 2},
            {2, 2},
            {1, 28, 28, 1}
        }
    };
}

std::vector<TestSuite::DenseTestCase> TestSuite::get_dense_test_cases() {
    return {
        {
            "Basic dense with ReLU",
            {64, 32, "relu"},
            1
        },
        {
            "Dense with Softmax",
            {128, 10, "softmax"},
            4
        },
        {
            "Large dense layer",
            {512, 256, "relu"},
            8
        },
        {
            "Dense with custom weights",
            {64, 32, "relu"},
            1
        },
        {
            "Dense with custom biases",
            {64, 32, "relu"},
            1
        }
    };
}

// Convolution layer tests
std::map<std::string, TestReport> TestSuite::test_convolution_layer() {
    std::map<std::string, TestReport> report;
    auto test_cases = get_conv_test_cases();

    for (const auto& case_info : test_cases) {
        std::cout << "\nRunning test: " << case_info.name << std::endl;

        ConvolutionLayer layer(
            case_info.params.num_filters,
            case_info.params.kernel_size,
            case_info.params.input_channels
        );

        auto test_input = xt::random::randn<double>(case_info.input_shape);
        auto expected_shape = layer.get_output_shape(case_info.input_shape, case_info.padding);
        auto output = layer.forward(test_input, case_info.padding);

        TestReport test_report;
        test_report.input_shape = std::vector<size_t>(test_input.shape().begin(), test_input.shape().end());
        test_report.output_shape = std::vector<size_t>(output.shape().begin(), output.shape().end());
        test_report.expected_shape = expected_shape;
        test_report.min_output = xt::amin(output)();
        test_report.max_output = xt::amax(output)();
        test_report.mean_output = calculate_mean(output);

        // Assertions
        assert(output.shape() == xt::xarray<double>::shape_type(expected_shape.begin(), expected_shape.end()));
        assert(!xt::any(xt::isnan(output)));
        assert(!xt::any(xt::isinf(output)));
        assert(xt::all(output >= 0));

        std::cout << "✓ Test passed" << std::endl;
        report[case_info.name] = test_report;
    }

    return report;
}

// Max pooling layer tests
std::map<std::string, TestReport> TestSuite::test_max_pooling_layer() {
    std::map<std::string, TestReport> report;
    auto test_cases = get_pool_test_cases();

    for (const auto& case_info : test_cases) {
        std::cout << "\nRunning test: " << case_info.name << std::endl;

        MaxPoolingLayer layer(
            case_info.pool_size,
            case_info.strides
        );


        auto test_input = xt::random::randn<double>(case_info.input_shape);
        auto expected_shape = layer.get_output_shape(case_info.input_shape);
        auto output = layer.forward(test_input);

        TestReport test_report;
        test_report.input_shape = std::vector<size_t>(test_input.shape().begin(), test_input.shape().end());
        test_report.output_shape = std::vector<size_t>(output.shape().begin(), output.shape().end());
        test_report.expected_shape = expected_shape;
        test_report.min_output = xt::amin(output)();
        test_report.max_output = xt::amax(output)();
        test_report.mean_output = calculate_mean(output);

        // Assertions
        assert(output.shape() == xt::xarray<double>::shape_type(expected_shape.begin(), expected_shape.end()));
        assert(!xt::any(xt::isnan(output)));
        assert(!xt::any(xt::isinf(output)));
        assert(xt::all(output >= 0));

        std::cout << "✓ Test passed" << std::endl;
        report[case_info.name] = test_report;
    }

    return report;
}

// Dense layer tests
std::map<std::string, TestReport> TestSuite::test_dense_layer() {
    std::map<std::string, TestReport> report;
    auto test_cases = get_dense_test_cases();

    for (const auto& case_info : test_cases) {
        std::cout << "\nRunning test: " << case_info.name << std::endl;


        DenseLayer layer(
            case_info.params.input_size,
            case_info.params.output_size,
            case_info.params.activation
        );

        auto test_input = xt::random::randn<double>({case_info.batch_size, case_info.params.input_size});
        auto output = layer.forward(test_input);

        TestReport test_report;
        test_report.input_shape = std::vector<size_t>(test_input.shape().begin(), test_input.shape().end());
        test_report.output_shape = std::vector<size_t>(output.shape().begin(), output.shape().end());
        test_report.expected_shape = {case_info.batch_size, case_info.params.output_size};
        test_report.min_output = xt::amin(output)();
        test_report.max_output = xt::amax(output)();
        test_report.mean_output = calculate_mean(output);

        // Assertions
        assert(output.shape() == xt::xarray<double>::shape_type(test_report.expected_shape.begin(), test_report.expected_shape.end()));
        assert(!xt::any(xt::isnan(output)));
        assert(!xt::any(xt::isinf(output)));
        assert(xt::all(output >= 0));

        std::cout << "✓ Test passed" << std::endl;
        report[case_info.name] = test_report;
    }

    return report;
}