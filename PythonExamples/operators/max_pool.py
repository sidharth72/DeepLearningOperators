import numpy as np

class MaxPoolingLayer:
    def __init__(self, pool_size=(2, 2), strides=None):
        """
        Initialize MaxPooling layer.
        
        Args:
            pool_size (tuple): Size of pooling window (height, width)
            strides (tuple): Stride size. If None, same as pool_size
        """
        self.pool_height, self.pool_width = pool_size
        self.strides = strides if strides is not None else pool_size
        self.stride_height, self.stride_width = self.strides

    def forward(self, input_data):
        batch_size, input_height, input_width, channels = input_data.shape
        output_height = (input_height - self.pool_height) // self.stride_height + 1
        output_width = (input_width - self.pool_width) // self.stride_width + 1

        output = np.zeros((batch_size, output_height, output_width, channels))

        for b in range(batch_size):
            for h in range(output_height):
                h_start = h * self.stride_height
                h_end = h_start + self.pool_height

                for w in range(output_width):
                    w_start = w * self.stride_width
                    w_end = w_start + self.pool_width

                    for c in range(channels):
                        window = input_data[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.max(window)

        return output

    def get_output_shape(self, input_shape):
        batch_size, input_height, input_width, channels = input_shape
        output_height = (input_height - self.pool_height) // self.stride_height + 1
        output_width = (input_width - self.pool_width) // self.stride_width + 1
        return (batch_size, output_height, output_width, channels)