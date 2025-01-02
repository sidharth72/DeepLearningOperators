import numpy as np

class DenseLayer:
    def __init__(self, weights_path=None, biases_path=None, test_params=None, activation='relu'):
        """
        Initialize Dense layer either with pretrained weights or test parameters.
        
        Args:
            weights_path (str): Path to saved weights
            biases_path (str): Path to saved biases
            test_params (dict): Parameters for test mode including:
                - input_size: Size of input features
                - output_size: Size of output features
                - activation: Activation function ('relu' or 'softmax')
        """
        if weights_path and biases_path:
            self.weights = np.load(weights_path)
            self.biases = np.load(biases_path)
            self.input_size = self.weights.shape[0]
            self.output_size = self.weights.shape[1]
        elif test_params:
            self.input_size = test_params['input_size']
            self.output_size = test_params['output_size']
            self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
            self.biases = np.zeros(self.output_size)
        else:
            raise ValueError("Either provide paths to weights or test parameters")
            
        self.activation = activation

    def _relu(self, x):
        return np.maximum(x, 0)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, input_data):
        self._validate_input(input_data)
        linear_output = np.dot(input_data, self.weights) + self.biases

        if self.activation == 'relu':
            return self._relu(linear_output)
        elif self.activation == 'softmax':
            return self._softmax(linear_output)
        else:
            return linear_output

    def _validate_input(self, input_data):
        if len(input_data.shape) != 2:
            raise ValueError(
                f"Expected 2D input tensor, got shape {input_data.shape}"
            )
            
        if input_data.shape[1] != self.input_size:
            raise ValueError(
                f"Input size {input_data.shape[1]} doesn't match weight matrix "
                f"input size {self.input_size}"
            )
