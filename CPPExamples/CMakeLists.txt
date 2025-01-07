# =================================================================
# Project Initialization
# =================================================================
cmake_minimum_required(VERSION 3.13)

# Set vcpkg toolchain file - IMPORTANT: This must come before project()
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
else()
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(CNNOperators)

# Configure C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add this line to help find OpenCV
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/vcpkg/installed/x64-windows")

# =================================================================
# Project Structure and Directory Setup
# =================================================================
set(OPERATORS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/operators")
set(TEST_OPERATORS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_operators")
set(MODEL_PARAMETER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/models_and_parameters")

# =================================================================
# Find Required Packages
# =================================================================
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Manual ONNXRuntime Configuration
set(ONNXRUNTIME_ROOT "C:/vcpkg/installed/x64-windows/include/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIRS "${ONNXRUNTIME_ROOT}/lib")

if(NOT DEFINED onnxruntime_FOUND)
    message(STATUS "Manual ONNXRuntime configuration")
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
        set(onnxruntime_FOUND TRUE)
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIRS}/onnxruntime.dll"
            IMPORTED_IMPLIB "${ONNXRUNTIME_LIB_DIRS}/onnxruntime.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
        )
        message(STATUS "ONNXRuntime include dir: ${ONNXRUNTIME_INCLUDE_DIRS}")
        message(STATUS "ONNXRuntime library dir: ${ONNXRUNTIME_LIB_DIRS}")
    else()
        message(FATAL_ERROR "ONNXRuntime headers not found in ${ONNXRUNTIME_INCLUDE_DIRS}")
    endif()
endif()

find_package(OpenBLAS CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)

# =================================================================
# Common Include Directories
# =================================================================
set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPERATORS_DIR}/include
    ${TEST_OPERATORS_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include
)

# =================================================================
# Utility Libraries
# =================================================================
add_library(model_loader 
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities/ModelLoader.cpp
)
target_include_directories(model_loader PUBLIC 
    ${COMMON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities
)
target_link_libraries(model_loader PUBLIC
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# =================================================================
# Neural Network Operators
# =================================================================
function(add_operator_library name source)
    add_library(${name} ${OPERATORS_DIR}/src/${source})
    target_include_directories(${name} PUBLIC ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${name} PUBLIC
        xtensor
        xtensor-blas
        OpenBLAS::OpenBLAS
        nlohmann_json::nlohmann_json
    )
endfunction()

add_operator_library(convolution_layer "ConvolutionLayer.cpp")
add_operator_library(maxpooling_layer "MaxPoolingLayer.cpp")
add_operator_library(flatten_layer "FlattenLayer.cpp")
add_operator_library(dense_layer "DenseLayer.cpp")
add_operator_library(softmax_layer "SoftmaxLayer.cpp")
add_operator_library(relu_layer "ReLULayer.cpp")
add_operator_library(batch_normalization_layer "BatchNormalizationLayer.cpp")

# =================================================================
# Main Executable Configuration
# =================================================================
add_executable(main src/main.cpp)
add_executable(tests src/tests.cpp 
    test_operators/unit_tests/layer_tests.cpp
    test_operators/model_tests/model_tests.cpp
)   
add_executable(onnx_inference src/onnx_inference.cpp)

# Link libraries for main executable
target_link_libraries(main PRIVATE
    convolution_layer
    maxpooling_layer
    flatten_layer
    dense_layer
    relu_layer
    softmax_layer
    batch_normalization_layer
    model_loader
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    onnxruntime
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
)

# Link libraries for tests
target_link_libraries(tests PRIVATE 
    convolution_layer
    maxpooling_layer
    flatten_layer
    dense_layer
    relu_layer
    softmax_layer
    batch_normalization_layer
    model_loader
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    onnxruntime
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS

)

# Link libraries for onnx_inference
target_link_libraries(onnx_inference PRIVATE
    model_loader
    ${OpenCV_LIBS}
    onnxruntime
    nlohmann_json::nlohmann_json
)

# Add test includes
target_include_directories(tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/test_operators/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(onnx_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities
    ${COMMON_INCLUDE_DIRS}
)

# =================================================================
# Configuration Summary
# =================================================================
function(print_status_message)
    message(STATUS "Configuration Summary:")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "nlohmann_json found: ${nlohmann_json_FOUND}")
    message(STATUS "onnxruntime found: ${onnxruntime_FOUND}")
    message(STATUS "xtensor found: ${xtensor_FOUND}")
    message(STATUS "xtensor-blas found: ${xtensor-blas_FOUND}")
    message(STATUS "OpenBLAS found: ${OpenBLAS_FOUND}")
endfunction()

print_status_message()