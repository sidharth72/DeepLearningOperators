import numpy as np
from operators.conv import ConvolutionLayer
from operators.max_pool import MaxPoolingLayer
from operators.dense import DenseLayer
from operators.flatten import FlattenLayer
from test_operators.unit_tests.layers_tests import test_convolution_layer, test_max_pooling, test_dense
from test_operators.model_tests.model_tests import test_model_layers
from main import ModelPredictor
from datetime import datetime
from pathlib import Path
import time
import json

def append_to_report(report_data, test_type = 'layer'):
    """
    Append a new test report to the JSON file while maintaining all previous reports.
    Creates the reports directory and file if they don't exist.
    """
    # Create reports directory if it doesn't exist
    reports_dir = Path("report")
    reports_dir.mkdir(exist_ok=True)

    report = None

    if test_type == 'layer':
        report_file = reports_dir / "model_unittest_report.json"
    else:

        report_file = reports_dir / "model_test_report.json"
    
    # Read existing reports if file exists
    existing_reports = []
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                existing_reports = json.load(f)
                if not isinstance(existing_reports, list):
                    existing_reports = [existing_reports]
        except json.JSONDecodeError:
            # If file is corrupted, start fresh but don't lose the file
            existing_reports = []
    
    # Append new report
    existing_reports.append(report_data)
    
    # Save all reports atomically
    temp_file = report_file.with_suffix('.tmp')
    try:
        with open(temp_file, 'w') as f:
            json.dump(existing_reports, f, indent=4)
        # Atomic replacement of the file
        temp_file.replace(report_file)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise e

def layer_tests(layer_type):
    start_time = time.time()
    
    try:
        if layer_type == 'conv':
            test_results = test_convolution_layer(ConvolutionLayer)
        elif layer_type == 'maxpool':
            test_results = test_max_pooling(MaxPoolingLayer)
        elif layer_type == 'dense':
            test_results = test_dense(DenseLayer)
        else:
            raise ValueError('Invalid layer type')
        
        # Create simple report
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'layer_type': layer_type,
            'tests_passed': True,
            'execution_time': f"{time.time() - start_time:.2f} seconds",
            'results': test_results
        }
        
        # Append to report file
        append_to_report(report)
        
        print(f"\nTest completed successfully!")
        print(f"Layer Type: {layer_type}")
        print(f"Execution Time: {report['execution_time']}")
        print("Results saved to reports/model_unittest_report.json")
        
        return report
        
    except Exception as e:
        error_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'layer_type': layer_type,
            'tests_passed': False,
            'error': str(e),
            'execution_time': f"{time.time() - start_time:.2f} seconds"
        }
        
        append_to_report(error_report)
        print(f"\nTest failed!")
        print(f"Layer Type: {layer_type}")
        print(f"Error: {str(e)}")
        print(f"Execution Time: {error_report['execution_time']}")
        print("Results saved to reports/model_unittest_report.json")
        
        return error_report

if __name__ == '__main__':
    # Get test type from user, either Layer wise tests or Model wise tests
    test_type = input("Enter test type: Type 'L' for Layer wise tests, 'M' for Model wise tests: ")
    if test_type == 'L':
        layer_type = input("Enter layer type: Type 'conv' for Convolutional layer, 'maxpool' for MaxPooling layer, 'dense' for Dense layer: ")
        report = layer_tests(layer_type)
        # Remove the file overwrite here since we're already appending in layer_tests()

    elif test_type == 'M':
        image_path = input("Enter image path: (path/to/image.jpg)")
        class_label = input(
            """
airplane,
automobile,
bird,
cat,
deer,
dog,
frog,
horse,
ship,
truck

Enter expected class label:
"""
        )
        config_file_path = "config/network_config.json"
        predictor = ModelPredictor(config_file_path)
        model_layers = predictor.load_model(predictor.models_config[2])
        input_data = predictor.preprocess_image(image_path)
        excepted_label = predictor.global_settings['classes'].index(class_label)

        results = test_model_layers(
            layers=model_layers,
            input_data=input_data,
            expected_label=excepted_label,
            class_labels=predictor.global_settings['classes']
        )

        # Add model test results to the report as well
        model_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'test_type': 'model',
            'tests_passed': bool(results['test_passed']),
            'prediction_correct': bool(results.get('prediction_correct', False)),
            'error': str(results.get('error', None))
        }
        append_to_report(model_report, test_type='model')

        if results['test_passed']:
            print("\nAll test cases passed!")
            if results['prediction_correct']:
                print("Prediction matches expected label!")
            else:
                print("Prediction does not match expected label.")
        else:
            print("\nTest failed:", results['error'])

    else:
        print("Invalid test type")