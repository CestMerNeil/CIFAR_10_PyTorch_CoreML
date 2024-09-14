
# CIFAR-10 PyTorch to CoreML Conversion

This project demonstrates how to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch, export the trained model to CoreML format, and use the model for inference in a macOS environment via Swift.

## Features
- Train a CNN on CIFAR-10 dataset with PyTorch.
- Export the trained PyTorch model to CoreML format.
- Use CoreML model in a macOS command-line interface using Swift.

## Requirements

### Python Requirements
- Python 3.x
- PyTorch
- torchvision
- CoreMLTools

Install dependencies via pip:
```bash
pip install torch torchvision coremltools
```

### macOS Requirements
- Xcode 12+
- Swift 5.0+
- macOS 11.0+

## Usage

### Model Training and Conversion
1. Train the PyTorch model using the provided `TorchModelTrain.ipynb` notebook.
2. Export the trained model to CoreML using CoreMLTools:
   ```python
   import coremltools as ct
   coreml_model = ct.convert(trained_model)
   coreml_model.save('CIFAR10Net.mlmodel')
   ```

### Swift Integration
1. Integrate the `.mlmodel` into your macOS project.
2. Use the model in Swift as shown in the example Swift code:
   ```swift
   let model = CIFAR10NetMPS_4()
   // Use the model for inference
   ```

## Project Structure
- `TorchModelTrain.ipynb`: Jupyter notebook for training and converting the model.
- `CIFAR10NetMPS_4.mlpackage`: CoreML model package for macOS.
- `main.swift`: Example Swift code for using the CoreML model.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
