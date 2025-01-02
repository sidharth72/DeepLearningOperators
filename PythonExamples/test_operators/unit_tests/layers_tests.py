# Test the architecure of the model with multiple dimensions, filter sizes, etc.
# Test multiple input sizes, dense layer sizes, etc.
# Test models layer output in comparison to expected output.

# ConvolutionLayer test cases
# testcase 1: kernel 3x3, padding same
# testcase 2: kernel 5x5, padding valid
# testcase 3: kernel 3x3, padding valid
# testcase 4: kernel 5x5, padding same
# testcase 5: kernel 7x7, padding same

# DenseLayer test cases
# testcase 1: hidden layer size 64, activation relu
# testcase 2: hidden layer size 128, activation relu
# testcase 3: hidden layer size 256, activation relu
# testcase 4: activation relu, activation softmax
# testcase 5: compare difference between custom layer output and keras output

# MaxPoolingLayer test cases
# testcase 1: pool size 2x2, stride 2x2
# testcase 2: pool size 3x3, stride 3x3
# testcase 3: pool size 2x2, stride 1x1
# testcase 4: pool size 3x3, stride 1x1
# testcase 5: compare difference between custom layer output and keras output

import numpy as np

def test_convolution_layer(ConvolutionLayer):
    test_cases = [
        {
            'name': 'Basic test with same padding',
            'params': {
                'num_filters': 32,
                'kernel_size': 3,
                'input_channels': 3
            },
            'input_shape': (1, 32, 32, 3),
            'padding': 'same'
        },
        {
            'name': 'Larger input with valid padding',
            'params': {
                'num_filters': 64,
                'kernel_size': 5,
                'input_channels': 3
            },
            'input_shape': (4, 64, 64, 3),
            'padding': 'valid'
        },
        {
            'name': 'Single channel input',
            'params': {
                'num_filters': 16,
                'kernel_size': 3,
                'input_channels': 1
            },
            'input_shape': (2, 28, 28, 1),
            'padding': 'same'
        },
        {
            'name': 'Large kernel with valid padding',
            'params': {
                'num_filters': 32,
                'kernel_size': 5,
                'input_channels': 3
            },
            'input_shape': (1, 64, 64, 3),
            'padding': 'valid'
        },
        {
            'name': 'Large kernel with same padding',
            'params': {
                'num_filters': 64,
                'kernel_size': 7,
                'input_channels': 3
            },
            'input_shape': (1, 64, 64, 3),
            'padding': 'same'
        }

    ]

    report = {}

    for case in test_cases:
        print(f"\nRunning test: {case['name']}")
        
        # Create layer
        layer = ConvolutionLayer(test_params=case['params'])
        
        # Generate random input
        test_input = np.random.randn(*case['input_shape'])
        
        # Get expected output shape
        expected_shape = layer.get_output_shape(case['input_shape'], case['padding'])
        
        # Run forward pass
        output = layer.forward(test_input, padding=case['padding'])
        
        # Print results
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        print(f"Min output value: {output.min():.4f}")
        print(f"Max output value: {output.max():.4f}")
        print(f"Mean output value: {output.mean():.4f}")

        # Saving the report
        report[case['name']] = {
            'input_shape': test_input.shape,
            'output_shape': output.shape,
            'expected_shape': expected_shape,
            'min_output': output.min(),
            'max_output': output.max(),
            'mean_output': output.mean()
        }
        
        # Basic assertions
        assert output.shape == expected_shape, f"Shape mismatch: got {output.shape}, expected {expected_shape}"
        assert not np.any(np.isnan(output)), "Output contains NaN values"
        assert not np.any(np.isinf(output)), "Output contains infinity values"
        assert np.all(output >= 0), "ReLU activation failed - negative values in output"
        
        print("✓ Test passed")

    return report


def test_max_pooling(MaxPoolingLayer):
    test_cases = [
        {
            'name': 'Basic 2x2 pooling with default stride',
            'pool_size': (2, 2),
            'strides': None,
            'input_shape': (1, 4, 4, 3)
        },
        {
            'name': 'Custom pooling with different stride',
            'pool_size': (3, 3),
            'strides': (2, 2),
            'input_shape': (2, 8, 8, 1)
        },
        {
            'name': 'Large input with default pooling',
            'pool_size': (2, 2),
            'strides': None,
            'input_shape': (4, 32, 32, 64)
        },
        {
            'name': 'Custom pooling with same stride',
            'pool_size': (3, 3),
            'strides': (1, 1),
            'input_shape': (1, 16, 16, 3)
        },
        {
            'name': 'Single channel input',
            'pool_size': (2, 2),
            'strides': None,
            'input_shape': (1, 28, 28, 1)
        }

    ]

    report = {}

    for case in test_cases:
        print(f"\nRunning MaxPooling test: {case['name']}")
        
        # Create layer
        layer = MaxPoolingLayer(
            pool_size=case['pool_size'],
            strides=case['strides']
        )
        
        # Generate random input
        test_input = np.random.randn(*case['input_shape'])
        
        # Get expected output shape
        expected_shape = layer.get_output_shape(case['input_shape'])
        
        # Run forward pass
        output = layer.forward(test_input)
        
        # Print results
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        print(f"Max input value: {test_input.max():.4f}")
        print(f"Max output value: {output.max():.4f}")

        # Saving the report
        report[case['name']] = {
            'input_shape': test_input.shape,
            'output_shape': output.shape,
            'expected_shape': expected_shape,
            'max_input': test_input.max(),
            'max_output': output.max()
        }
        
        # Verify that output values are maximum values from their respective windows
        assert output.shape == expected_shape, f"Shape mismatch: got {output.shape}, expected {expected_shape}"
        assert np.all(output <= test_input.max()), "Output values exceed input maximum"
        
        print("✓ Test passed")

    return report

def test_dense(DenseLayer):
    test_cases = [
        {
            'name': 'Basic dense with ReLU',
            'params': {
                'input_size': 64,
                'output_size': 32,
                'activation': 'relu'
            },
            'batch_size': 1
        },
        {
            'name': 'Dense with Softmax',
            'params': {
                'input_size': 128,
                'output_size': 10,
                'activation': 'softmax'
            },
            'batch_size': 4
        },
        {
            'name': 'Large dense layer',
            'params': {
                'input_size': 512,
                'output_size': 256,
                'activation': 'relu'
            },
            'batch_size': 8
        },
        {
            'name': 'Dense with custom weights',
            'params': {
                'input_size': 64,
                'output_size': 32,
                'activation': 'relu'
            },
            'batch_size': 1
        },
        {
            'name': 'Dense with custom biases',
            'params': {
                'input_size': 64,
                'output_size': 32,
                'activation': 'relu'
            },
            'batch_size': 1
        }

    ]

    report = {}

    for case in test_cases:
        print(f"\nRunning Dense test: {case['name']}")
        
        # Create layer
        layer = DenseLayer(test_params=case['params'], activation=case['params']['activation'])
        
        # Generate random input
        input_shape = (case['batch_size'], case['params']['input_size'])
        test_input = np.random.randn(*input_shape)
        
        # Run forward pass
        output = layer.forward(test_input)
        
        # Print results
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Min output value: {output.min():.4f}")
        print(f"Max output value: {output.max():.4f}")
        print(f"Mean output value: {output.mean():.4f}")

        # Saving the report
        report[case['name']] = {
            'input_shape': test_input.shape,
            'output_shape': output.shape,
            'min_output': output.min(),
            'max_output': output.max(),
            'mean_output': output.mean()
        }
        
        # Basic assertions
        expected_shape = (case['batch_size'], case['params']['output_size'])
        assert output.shape == expected_shape, f"Shape mismatch: got {output.shape}, expected {expected_shape}"
        
        if case['params']['activation'] == 'relu':
            assert np.all(output >= 0), "ReLU activation failed - negative values in output"
        elif case['params']['activation'] == 'softmax':
            assert np.allclose(np.sum(output, axis=1), 1.0), "Softmax probabilities don't sum to 1"
            assert np.all(output >= 0) and np.all(output <= 1), "Softmax values outside [0,1] range"
        
        print("✓ Test passed")

    return report
