# Neural-Network-using-Pytorch

This repository contains three different models for multi-class classification using PyTorch, all trained on the Fashion MNIST dataset. The models include an Artificial Neural Network (ANN), a custom Convolutional Neural Network (CNN), and a CNN model with pre-trained ResNet weights. The goal of the project is to practice neural network design and enhance model performance using various techniques.

## Models

### 1. **Artificial Neural Network (ANN)**
   - **Input size**: 784 (28x28 flattened pixels from Fashion MNIST)
   - **Architecture**: 
     - Input → Linear (784 → 128) → BatchNorm → ReLU → Dropout
     - Linear (128 → 64) → BatchNorm → ReLU → Dropout
     - Output → Linear (64 → 10)
   - **Features**:
     - Batch normalization and dropout to speed up learning and reduce overfitting.
     - Predicts 10 classes (Fashion MNIST categories).
  
### 2. **Custom Convolutional Neural Network (CNN)**
   - **Input size**: 28x28 image (no flattening required)
   - **Architecture**: 
     - Feature Extraction: 2 Convolutional layers with ReLU activation and BatchNorm
     - Max Pooling layers
     - Fully connected classifier with Dropout for regularization
   - **Features**:
     - 1st Convolution Layer: (1 → 32 channels)
     - 2nd Convolution Layer: (32 → 64 channels)
     - Output layer with 10 classes (Fashion MNIST categories).
     - Batch normalization and dropout used for better learning and complexity reduction.

### 3. **CNN with Pre-trained ResNet Weights**
   - **Input size**: 28x28 image (no flattening required)
   - **Feature Extraction**: Pre-trained ResNet layers with frozen parameters
   - **Classification**: Custom fully connected layer (fine-tuned)
   - **Architecture**:
     - **Feature extraction**: Pre-trained ResNet (frozen layers)
     - **Classification**: Linear (2048 → 128) → ReLU → Dropout → Linear (128 → 64) → ReLU → Dropout → Linear (64 → 10)
   - **Features**:
     - Transfer learning with frozen feature extraction layers.
     - Fine-tuning of the classification layer to adapt to the Fashion MNIST dataset.

## Performance

- **ANN**: 
   - Training Accuracy: 99% 
   - Test Accuracy: 82%
   - Shows overfitting (higher training accuracy, lower test accuracy).
  
- **Custom CNN**:
   - Training Accuracy: 82%
   - Test Accuracy: 82%
  
- **CNN with Pre-trained ResNet**:
   - Training Accuracy: 87%
   - Test Accuracy: 87%
   - Better performance with higher epochs.

## Requirements
- PyTorch
- torchvision
- numpy
- matplotlib

## Installation

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

You can easily train and evaluate each of the models by running the respective Python files. The dataset will be automatically downloaded if not already available.

---

This repository provides a clear comparison of different neural network architectures on the Fashion MNIST dataset, highlighting the advantages of CNNs and transfer learning over traditional ANN models.
