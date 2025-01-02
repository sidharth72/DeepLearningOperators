import numpy as np

def test_model_layers(layers, input_data, expected_label, class_labels):
    """
    Test if data successfully passes through all layers and validates the prediction.
    
    Args:
        layers: List of tuples containing (layer_name, layer_object)
        input_data: Input data to test (should be preprocessed)
        expected_label: Expected class index
        class_labels: List of class names
    
    Returns:
        dict: Contains test results and model prediction
    """
    x = input_data
    all_layers_passed = True
    
    print("\nTesting model layers:")
    print("-" * 50)
    
    # Test each layer
    for layer_name, layer in layers:
        try:
            # Handle convolution layer padding
            if hasattr(layer, '__class__') and layer.__class__.__name__ == 'ConvolutionLayer':
                x = layer.forward(x, padding='same')
            else:
                x = layer.forward(x)
                
            print(f"✓ {layer_name} - Passed | Output shape: {x.shape}")
            
        except Exception as e:
            print(f"✗ {layer_name} - Failed: {str(e)}")
            all_layers_passed = False
            return {
                'test_passed': False,
                'error': f"Layer {layer_name} failed: {str(e)}",
                'prediction': None
            }
    
    # Get prediction
    prediction = np.argmax(x[0])
    confidence = float(x[0][prediction])
    predicted_class = class_labels[prediction]
    expected_class = class_labels[expected_label]
    
    print("\nTest Results:")
    print("-" * 50)
    print(f"All layers passed: {all_layers_passed}")
    print(f"Predicted class: {predicted_class}")
    print(f"Expected class: {expected_class}")
    print(f"Confidence: {confidence:.4f}")
    
    return {
        'test_passed': all_layers_passed,
        'prediction_correct': prediction == expected_label,
        'predicted_class': predicted_class,
        'expected_class': expected_class,
        'confidence': confidence
    }