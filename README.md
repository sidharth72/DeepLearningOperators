# Convolutional Neural Network Layer by Layer Operators From Scratch

This repository showcases Convolutional Neural Network models built from scratch in C++ and Python, trained on CIFAR-10. It focuses on understanding the core workings, layer-by-layer transformations, different operations, and how these models make predictions. The work also uses ONNX runtime to effectively inference the model.

---

## Table of Contents

- [Requirements](#requirements)
  - [General Requirements](#general-requirements)
  - [C++ Requirements](#c-requirements)
  - [Python Requirements](#python-requirements)
- [Installation](#installation)
  - [C++ Installation](#c-installation)
  - [Python Installation](#python-installation)
- [Building the Project](#building-the-project)
  - [Building the C++ Project](#building-the-c-project)
  - [Building the Python Project](#building-the-python-project)
- [Running the Project](#running-the-project)
  - [Running C++ Executables](#running-c-executables)
  - [Running Python Scripts](#running-python-scripts)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

### General Requirements

- CMake 3.13 or higher
- Git

### C++ Requirements

- Microsoft Visual Studio 2022 (or compatible C++ compiler)
- vcpkg (for managing C++ dependencies)
- OpenCV
- ONNX Runtime
- xtensor
- xtensor-blas
- nlohmann-json

### Python Requirements

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

---

## Installation

### C++ Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sidharth72/DeepLearningOperators.git
   cd DeepLearningOperators
   ```

2. **Install CMake:**

   - Download CMake from [CMake.org](https://cmake.org/download/).
   - Install and add it to your system PATH so it can be accessed via the command line.

3. **Install Microsoft Visual Studio 2022:**

   - Download and install [Visual Studio 2022](https://visualstudio.microsoft.com/vs/).
   - During installation, ensure you select the "Desktop development with C++" workload to install all necessary tools for modern C++ development.

4. **Configure the MSVC Compiler:**

   - Open the "Developer Command Prompt for VS" to ensure the environment variables for MSVC are automatically set.

5. **Install vcpkg:**

   ```bash
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh
   ```

6. **Install required libraries using vcpkg:**

   ```bash
   ./vcpkg install opencv nlohmann-json xtensor xtensor-blas
   ```

7. **Install ONNX Runtime Manually:**

   - Download ONNX Runtime from the official website: [ONNX Runtime Downloads](https://onnxruntime.ai/).
   - Extract the downloaded files.
   - Move the extracted files into the `<base_path>\vcpkg\installed\x64-windows\include` directory.
   - Rename the folder to `onnxruntime`.

8. **Organize External Dependencies:**

   - Ensure all dependencies (`xtensor`, `xtensor-blas`, `nlohmann-json`, `opencv`, and `onnxruntime`) are correctly stored in `<base_path>\vcpkg\installed\x64-windows\include`.

### Python Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sidharth72/DeepLearningOperators.git
   cd DeepLearningOperators
   ```

2. **Install Python dependencies:**

   ```bash
   cd PythonExamples
   pip install -r requirements.txt
   ```

---

## Building the Project

### Building the C++ Project

1. **Create a build directory:**

   ```bash
   cd CPPExamples
   mkdir build
   ```

2. **Configure the project using CMake:**

   ```bash
   cd build
   cmake .. -G "Visual Studio 17 2022" -A x64
   ```

3. **Build the project:**

   ```bash
   cmake --build . --config Release
   ```

### Building the Python Project

No build steps are required for the Python project. Ensure all dependencies are installed as mentioned in the [Python Installation](#python-installation) section.

---

## Running the Project

### Running C++ Executables

After building the project, you can run the executables located in the `build/Release` directory:

- **Main executable:** Runs the multiple CNN operators in an ensemble fashion

  ```bash
  cd build
  Release/main.exe
  ```

- **ONNX Inference:** Runs the inference using ONNX runtime.

  ```bash
  cd build
  Release/onnx_inference.exe path/to/image
  ```

- **Tests:** Runs the model unit and architecture tests.

  ```bash
  cd build
  Release/tests.exe
  ```

### Running Python Scripts

Navigate to the `PythonExamples` directory and run the scripts:

- **Main script:** Runs the neural network operators.

  ```bash
  python PythonExamples/main.py
  ```

- **Unit tests:** Runs the unit tests.

  ```bash
  python PythonExamples/tests.py
  ```

---

## Project Structure

```plaintext
CNNOperators/
├── .gitignore
├── .vscode/
│   ├── c_cpp_properties.json
│   ├── settings.json
│   ├── tasks.json
├── CPPExamples/
│   ├── .vs/
│   ├── build/
│   ├── config/
│   ├── data/
│   ├── external/
│   ├── include/
│   ├── models/
│   ├── operators/
│   ├── report/
│   ├── src/
│   ├── test_operators/
│   ├── utilities/
│   ├── CMakeLists.txt
├── PythonExamples/
│   ├── __pycache__/
│   ├── config/
│   ├── data/
│   ├── main.py
│   ├── models/
│   ├── operators/
│   ├── report/
│   ├── test_operators/
│   ├── tests.py
├── README.txt
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes
4. Commit your changes:
   ```bash
   git commit -am 'Add new feature'
   ```
5. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
6. Create a new Pull Request

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

