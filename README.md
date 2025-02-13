# MNIST Neural Network Classifier

This repository contains a PyTorch-based implementation for training and evaluating neural networks on the MNIST dataset. The project includes both a Feed-Forward Neural Network (FFNN) and a Convolutional Neural Network (CNN) for digit classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [License](#license)

## Project Overview

This project demonstrates the following:

- Loading and preprocessing the MNIST dataset.
- Implementing and training a Feed-Forward Neural Network (FFNN) for classification.
- Implementing and training a Convolutional Neural Network (CNN) for improved accuracy.
- Evaluating the trained models.
- Visualizing predictions and training progress.

## Installation

To run the project, ensure you have Python installed, along with the required libraries:

```sh
pip install torch torchvision matplotlib numpy
```

## Dataset

The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The dataset is automatically downloaded when running the script.

## Model Architectures

### Feed-Forward Neural Network (FFNN)

A simple neural network with:

- 784 input neurons (flattened 28x28 images)
- Two hidden layers with 128 and 64 neurons
- ReLU activation
- Output layer with 10 neurons using LogSoftmax activation

### Convolutional Neural Network (CNN)

A CNN model with:

- Three convolutional layers (32, 64, 128 filters)
- Batch normalization layers
- Fully connected layers with dropout
- LogSoftmax activation for classification

## Training

The training process involves:

1. Splitting the MNIST dataset into training and validation sets.
2. Defining hyperparameters such as learning rate and number of epochs.
3. Using stochastic gradient descent (SGD) optimization.
4. Computing the negative log-likelihood loss (NLLLoss).

To train the FFNN:

```python
train_model(model, optimizer, criterion, nepochs, train_loader, val_loader, is_image_input=True)
```

To train the CNN:

```python
train_model(new_cnn_model, cnn_optimizer, cnn_criterion, cnn_nepochs, train_loader, val_loader, is_image_input=False)
```

## Evaluation

Model accuracy is evaluated on the validation dataset:

```python
evaluate_model(model, val_loader, is_image_input=True)
```

For CNN evaluation:

```python
evaluate_model(new_cnn_model, val_loader, is_image_input=False)
```

## Visualization

- Randomly sample an image and visualize predictions:

```python
random_prediction_example(mnist_dataloader, model)
```

- Plot training and validation losses over epochs:

```python
plt.plot(range(1, nepochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, nepochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```
