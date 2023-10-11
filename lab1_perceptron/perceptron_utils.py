import numpy as np


class SigmoidActivation:
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of Sigmoid activation.

        Args:
            input (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the Sigmoid activation.
        """
        # Sigmoid function: f(x) = 1 / (1 + exp(-x))
        pass

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of Sigmoid activation.

        Args:
            input (np.ndarray): Input data.
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        # The derivative of the Sigmoid function:
        # f'(x) = f(x) * (1 - f(x))
        pass



class BinaryCrossEntropyLoss:
    def __init__(self, stability=1e-10) -> None:
        """
        Initialize the Binary Cross-Entropy Loss function.

        Args:
            stability (float, optional): A small constant to prevent division by zero in log calculations. Default is 1e-10.
        """
        self.stability = stability

    def forward(self, predictions: np.ndarray, target: np.ndarray) -> (float, np.ndarray):
        """
        Compute the forward pass of Binary Cross-Entropy loss.

        Args:
            predictions (np.ndarray): Model predictions, typically the output of a softmax layer.
            target (np.ndarray): Target values, which are the ground truth labels in a one-hot encoded format.

        Returns:
        tuple:
            float: The computed loss.
            np.ndarray: The gradient of the loss with respect to predictions.
        """
        pass

class SGD:
    def __init__(self, learning_rate, clip_range=10) -> None:
        """
        Initialize simple Stochastic Gradient Descent (SGD) optimizer.

        Args:
            learning_rate (float): The learning rate for weight and bias updates during optimization.
        """
        self.learning_rate = learning_rate

    def update_weights(self, weights: np.ndarray, weights_grad: np.ndarray) -> np.ndarray:
        """
        Update weights using Stochastic Gradient Descent (SGD).

        Args:
            weights (np.ndarray): Current weights of a model's parameters.
            weights_grad (np.ndarray): Gradient of the loss with respect to the weights.

        Returns:
            np.ndarray: Updated weights.
        """
        clipped_grad = np.clip(weights_grad, -self.clip_range, self.clip_range)
        pass
    
    def update_bias(self, bias: np.ndarray, bias_grad: np.ndarray) -> np.ndarray:
        """
        Update bias using Stochastic Gradient Descent (SGD).

        Args:
            bias (np.ndarray): Current bias of a model's parameters.
            bias_grad (np.ndarray): Gradient of the loss with respect to the bias.

        Returns:
            np.ndarray: Updated bias.
        """
        clipped_grad = np.clip(bias_grad, -self.clip_range, self.clip_range)
        pass

class Perceptron:
    def __init__(self, input_size, activation, optimizer) -> None:
        """
        Initialize Perceptron.

        Args:
            input_size (int): Number of input units.
            activation: The activation function to use (e.g., sigmoid, ReLU).
            optimizer: The optimizer for updating weights and biases (e.g., SGD, Adam).
        """
        self.input_size = input_size
        self.activation = activation
        self.optimizer = optimizer  

        #TODO: Initialize weights and biases
        self.weights = None
        self.bias = None
    
    def forward(self, input): 
        """
        Perform the forward pass through the perceptron.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output of the layer after applying activation, shape (batch_size, 1).
        """
        pass
    
    def backward(self, grad_output):
        """
        Perform the backward pass through the perceptron, updating it's weights.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            None
        """
        pass

