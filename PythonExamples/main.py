import json
import numpy as np
from PIL import Image
from operators.conv import ConvolutionLayer
from operators.max_pool import MaxPoolingLayer
from operators.flatten import FlattenLayer
from operators.dense import DenseLayer
import os

class ModelPredictor:
    def __init__(self, config_path):
        """Initialize predictor with configuration file"""
        with open(config_path, 'r') as f:
            self.full_config = json.load(f)
        
        self.global_settings = self.full_config['global_settings']
        self.models_config = self.full_config['models']
        
    def load_model(self, model_config):
        """Load a single model configuration and create layers"""
        layers = []
        base_dir = self.global_settings['base_weights_directory']
        model_dir = model_config['model_directory']
        
        for layer_config in model_config['layers']:
            if layer_config['type'] == 'convolution':
                filters_path = os.path.join(base_dir, model_dir, layer_config['weights']['filters'])
                biases_path = os.path.join(base_dir, model_dir, layer_config['weights']['biases'])
                
                layer = ConvolutionLayer(
                    filters_path,
                    biases_path
                )
            elif layer_config['type'] == 'maxpooling':
                layer = MaxPoolingLayer(
                    pool_size=layer_config['parameters']['pool_size']
                )
            elif layer_config['type'] == 'flatten':
                layer = FlattenLayer()
            elif layer_config['type'] == 'dense':
                weights_path = os.path.join(base_dir, model_dir, layer_config['weights']['weights'])
                biases_path = os.path.join(base_dir, model_dir, layer_config['weights']['biases'])
                layer = DenseLayer(
                    weights_path,
                    biases_path,
                    activation = layer_config['parameters'].get('activation', 'relu')
                )
            layers.append((layer_config['name'], layer))
        
        return layers
    
    def preprocess_image(self, image_path):
        """Preprocess input image according to global settings"""
        img = Image.open(image_path)
        img = img.resize(tuple(self.global_settings['preprocessing']['input_size']))
        img_arr = np.array(img, dtype=np.float32)
        
        if self.global_settings['preprocessing']['normalization']['type'] == 'divide':
            img_arr = img_arr / self.global_settings['preprocessing']['normalization']['value']
        
        return np.expand_dims(img_arr, axis=0)
    
    def predict_single_model(self, input_data, model_config):
        """Make prediction using a single model"""
        # Load model layers
        layers = self.load_model(model_config)
        
        # Forward pass through the network
        x = input_data
        for i, (name, layer) in enumerate(layers):
            if isinstance(layer, ConvolutionLayer):
                # Get padding from the layer's configuration
                padding = model_config['layers'][i]['parameters']['padding']
                x = layer.forward(x, padding=padding)
            else:
                x = layer.forward(x)
            print(f"{model_config['model_name']} - {name} output shape: {x.shape}")
        
        # Get prediction
        prediction = np.argmax(x)
        confidence = float(x[0][prediction])
        predicted_class = self.global_settings['classes'][prediction]
        
        return {
            'model_name': model_config['model_name'],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'raw_output': x[0]
        }
    
    def predict_all_models(self, image_path):
        """Make predictions using all configured models"""
        # Preprocess input image once
        input_data = self.preprocess_image(image_path)
        # Run prediction for each model
        predictions = []
        for model_config in self.models_config:
            try:
                print(f"\nProcessing model: {model_config['model_name']}")
                prediction = self.predict_single_model(input_data, model_config)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error processing model {model_config['model_name']}: {str(e)}")
                continue
        
        return predictions
    
    def ensemble_predictions(self, predictions):
        """Combine predictions from multiple models using simple averaging"""
        if not predictions:
            return None
            
        # Stack all raw outputs
        all_outputs = np.stack([pred['raw_output'] for pred in predictions])
        
        # Average the predictions
        avg_output = np.mean(all_outputs, axis=0)
        ensemble_prediction = np.argmax(avg_output)
        ensemble_confidence = float(avg_output[ensemble_prediction])
        
        return {
            'model_name': 'ensemble',
            'predicted_class': self.global_settings['classes'][ensemble_prediction],
            'confidence': ensemble_confidence
        }

def main():
    config_path = "config/network_config.json"
    image_path = "data/inputs/cat2.jpg"
    
    # Initialize predictor
    predictor = ModelPredictor(config_path)
    
    # Get predictions from all models
    predictions = predictor.predict_all_models(image_path)
    
    # Print individual model predictions
    print("\nIndividual Model Predictions:")
    print("-" * 50)
    for pred in predictions:
        print(f"Model: {pred['model_name']}")
        print(f"Predicted class: {pred['predicted_class']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print("-" * 50)
    
    # Get and print ensemble prediction
    ensemble_pred = predictor.ensemble_predictions(predictions)
    if ensemble_pred:
        print("\nEnsemble Prediction:")
        print("-" * 50)
        print(f"Predicted class: {ensemble_pred['predicted_class']}")
        print(f"Confidence: {ensemble_pred['confidence']:.4f}")

if __name__ == "__main__":
    main()