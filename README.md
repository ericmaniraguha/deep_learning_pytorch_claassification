# MNIST Digit Classification with MLP Neural Network

## Introduction

This repository contains a Python script for training and evaluating a Multi-Layer Perceptron (MLP) neural network on the MNIST dataset for digit classification. The script leverages PyTorch for model creation and training.

## Features

1. **MLP Model Definition:** The script defines an MLP neural network with three layers: an input layer with 28x28 units, a hidden layer with 128 units, another hidden layer with 64 units, and an output layer with 10 units, suitable for classifying digits from 0 to 9.

2. **Data Loading and Preprocessing:** It loads the MNIST dataset, performs data transformations, and creates data loaders for both training and testing datasets. The training dataset is shuffled to improve training effectiveness.

3. **Display Sample Images:** The script includes a function to display a sample of training images with their corresponding labels, providing a visual understanding of the dataset.

4. **Training and Evaluation:** It trains the MLP model using mini-batch gradient descent with a specified number of epochs and learning rate. The Adam optimizer and cross-entropy loss are used. After training, it evaluates the model's accuracy on the testing dataset.

5. **Model Saving:** The trained model is saved to a file named 'mnist_mlp_trained_model.pth' for future use.

## Model Architecture

- **Input Layer:** 28x28
- **Hidden Layer 1:** 128 units
- **Hidden Layer 2:** 64 units
- **Output Layer:** 10 units

## Training Configuration

- **Learning Rate:** 0.001
- **Optimizer:** Adam

## Results

The trained MLP model achieved an accuracy of approximately 97.43% on the test dataset, demonstrating its effectiveness in classifying hand-written digits.

## Usage

To use this code, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   [git clone <repository_url>](https://github.com/ericmaniraguha/deep_learning_pytorch_claassification.git)
   
`pip install torch torchvision matplotlib` - Install the required libraries (PyTorch, torchvision, and matplotlib) using the following 
`python mnist_mlp.py` - Run the provided Python script to train, evaluate, and save the MLP model

## Conclusion
This repository provides a straightforward implementation of an MLP neural network for digit classification on the MNIST dataset. You can easily adapt and extend this code for other image classification tasks or explore various neural network architectures and hyperparameters to improve performance. 
