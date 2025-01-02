import numpy as np

class ConvolutionLayer:
    def __init__(self, filters_path=None, biases_path=None, test_params=None):
        """
        Initialize the ConvolutionLayer either with pretrained weights or test parameters.
        
        Args:
            filters_path (str): Path to saved filters for inference mode
            biases_path (str): Path to saved biases for inference mode
            test_params (dict): Parameters for test mode including:
                - num_filters: Number of filters
                - kernel_size: Size of the kernel (assumed square)
                - input_channels: Number of input channels
        """
        if filters_path and biases_path:
            self.filters = np.load(filters_path)
            self.biases = np.load(biases_path)
            print(f"Filter shape: {filters_path}", self.filters.shape)
        elif test_params:
            self.num_filters = test_params['num_filters']
            self.kernel_size = test_params['kernel_size']
            self.input_channels = test_params['input_channels']
            
            # Initialize random filters and biases
            self.filters = np.random.randn(
                self.kernel_size,
                self.kernel_size,
                self.input_channels,
                self.num_filters
            ) * 0.1
            
            self.biases = np.random.randn(self.num_filters) * 0.1
        else:
            raise ValueError("Either provide paths to weights or test parameters")
            
        self.num_filters = self.filters.shape[3]
        self.kernel_size = self.filters.shape[0]
        self.input_channels = self.filters.shape[2]

    def _pad_input(self, input_data, padding='same'):
        if padding == 'valid':
            return input_data

        pad_h = (self.kernel_size - 1) // 2
        pad_w = (self.kernel_size - 1) // 2

        return np.pad(
            input_data,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant'
        )

    def _convolve_single(self, input_slice, kernel):
        return np.sum(input_slice * kernel)

    def forward(self, input_data, padding='same'):
        """
        Forward pass of the convolution layer.
        """
        batch_size, input_height, input_width, _ = input_data.shape
        padded_input = self._pad_input(input_data, padding)
        
        if padding == 'same':
            output_height = input_height
            output_width = input_width
        else:
            output_height = input_height - self.kernel_size + 1
            output_width = input_width - self.kernel_size + 1

        output = np.zeros((batch_size, output_height, output_width, self.num_filters))

        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    for f in range(self.num_filters):
                        h_start = h
                        h_end = h + self.kernel_size
                        w_start = w
                        w_end = w + self.kernel_size

                        input_slice = padded_input[b, h_start:h_end, w_start:w_end, :]
                        conv_result = self._convolve_single(input_slice, self.filters[:, :, :, f])
                        output[b, h, w, f] = conv_result + self.biases[f]
        return np.maximum(output, 0)  # ReLU activation

    def get_output_shape(self, input_shape, padding='same'):
        batch_size, input_height, input_width, _ = input_shape
        
        if padding == 'same':
            return (batch_size, input_height, input_width, self.num_filters)
        else:  # 'valid'
            output_height = input_height - self.kernel_size + 1
            output_width = input_width - self.kernel_size + 1
            return (batch_size, output_height, output_width, self.num_filters)