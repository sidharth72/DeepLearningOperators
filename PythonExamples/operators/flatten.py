import numpy as np

class FlattenLayer:

    def __init__(self):
        self.input_shape = None

    def forward(self, input_data):
        self.input_shape = input_data.shape
        batch_size = self.input_shape[0]
        flattend_size = np.prod(self.input_shape[1:])
        output = input_data.reshape(batch_size, flattend_size)
        return output

    def get_output_shape(self):
        return (self.input_shape[0], np.prod(self.input_shape[1:]))

        