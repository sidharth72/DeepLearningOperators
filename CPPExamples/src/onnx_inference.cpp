#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>

// Modified loadImage function for CIFAR-10 (32x32 images)
static std::vector<float> loadImage(const std::string& filename, float normalize_value = 255.0f) {
    // Load image
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    // Resize image to CIFAR-10 dimensions (32x32)
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(32, 32));

    // Convert BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize
    cv::Mat float_img;
    rgb_image.convertTo(float_img, CV_32F, 1.0f / normalize_value);

    // Create the output vector with the correct size
    std::vector<float> output(3 * 32 * 32);
    
    // Convert to NCHW format
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 32; ++w) {
            cv::Vec3f pixel = float_img.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; ++c) {
                size_t idx = h * 32 * 3 + w * 3 + c;  // NHWC format
                output[idx] = pixel[c];
            }
        }
    }

    return output;
}

#include <chrono>
// ...existing code...

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* imagePath = argv[1];
    
    auto total_start = std::chrono::high_resolution_clock::now();

    // Define CIFAR-10 labels
    const std::vector<std::string> labels = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    auto modelPath = L"../models/ONNX_Models/model_kernel3x3_dense128.onnx";

    // Load and preprocess image with timing
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    std::vector<float> imageVec;
    try {
        imageVec = loadImage(imagePath, 255.0f);
    }
    catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        preprocess_end - preprocess_start).count();

    // Create session and run inference with timing
    auto inference_start = std::chrono::high_resolution_clock::now();
    
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cifar10-inference");
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    // CIFAR-10 specific constants
    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 32;
    constexpr int64_t height = 32;
    constexpr int64_t numClasses = 10;
    constexpr int64_t numInputElements = numChannels * height * width;

    if (imageVec.size() != numInputElements) {
        std::cout << "Invalid image format. Must be 32x32 RGB image." << std::endl;
        return 1;
    }

    // Create session options and load model
    Ort::SessionOptions ort_session_options;
    
    // Create CPU session
    try {
        session = Ort::Session(env, modelPath, ort_session_options);
    }
    catch (const Ort::Exception& e) {
        std::cout << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    // Define shapes for CIFAR-10
    const std::array<int64_t, 4> inputShape = { 1, height, width, numChannels }; // NCHW format
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    // Create input and output arrays
    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

    // Create input and output tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // Copy preprocessed image data to input array
    std::copy(imageVec.begin(), imageVec.end(), input.begin());

    // Get input and output names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };

    // Run inference
    try {
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (const Ort::Exception& e) {
        std::cout << "Inference error: " << e.what() << std::endl;
        return 1;
    }

    auto inference_end = std::chrono::high_resolution_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        inference_end - inference_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start).count();

    // Print timing results
    std::cout << "Timing Results:" << std::endl;
    std::cout << "Preprocessing time: " << preprocess_time << " ms" << std::endl;
    std::cout << "Inference time: " << inference_time << " ms" << std::endl;
    std::cout << "Total execution time: " << total_time << " ms" << std::endl;

    // Clean up allocated memory
    inputName.release();
    outputName.release();

    // Process and display results
    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    
    // Sort results by confidence
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), 
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    // Display top 5 predictions
    std::cout << "\nTop 5 predictions for CIFAR-10:\n";
    for (size_t i = 0; i < 5; ++i) {
        const auto& result = indexValuePairs[i];
        std::cout << i + 1 << ": " << labels[result.first] 
                  << " (confidence: " << result.second * 100.0f << "%)" << std::endl;
    }

    return 0;
}