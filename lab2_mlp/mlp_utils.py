import numpy as np

class ReLUActivation:
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of ReLU activation.

        Args:
            input (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the ReLU activation.
        """
        pass

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of ReLU activation.

        Args:
            input (np.ndarray): Input data.
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        pass

class SigmoidActivation:
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of Sigmoid activation.

        Args:
            input (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the Sigmoid activation.
        """
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
        pass

class SoftmaxActivation:
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of Softmax activation.

        Args:
            input (np.ndarray): Input data of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Output of the Softmax activation, with the same shape as input.
        """
        # Improve numerical stability by subtracting the maximum value from input.
        input = input - np.max(input, axis=1, keepdims=True)
        exponents = np.exp(input)
        
        pass

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of Softmax activation.

        Args:
            input (np.ndarray): Input data of shape (batch_size, num_classes).
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input, with the same shape as input.
        """
        pass


class CrossEntropyLoss:
    def __init__(self, stability=1e-10) -> None:
        """
        Initialize the Cross-Entropy Loss function.

        Args:
            stability (float, optional): A small constant to prevent division by zero in log calculations. Default is 1e-10.
        """
        self.stability = stability

    def forward(self, predictions: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the forward pass of Cross-Entropy loss.

        Args:
            predictions (np.ndarray): Model predictions, typically the output of a softmax layer.
            target (np.ndarray): Target values, which are the ground truth labels in a one-hot encoded format.

        Returns:
            float: The computed loss.
        """
        pass
    

class SGD:
    def __init__(self, learning_rate, clip_range=10) -> None:
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.

        Args:
            learning_rate (float): The learning rate for weight and bias updates during optimization.
        """
        self.learning_rate = learning_rate
        self.clip_range = clip_range

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
        updated_weights = weights - self.learning_rate * clipped_grad
        return updated_weights
    
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
        updated_bias = bias - self.learning_rate * clipped_grad
        return updated_bias

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation, optimizer) -> None:
        """
        Initialize a fully connected layer.

        Args:
            input_size (int): Number of input units.
            output_size (int): Number of output units.
            activation: The activation function to use (e.g., sigmoid, ReLU).
            optimizer: The optimizer for updating weights and biases (e.g., SGD, Adam).
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.optimizer = optimizer
        
        #TODO: Kaiming He initialization for weights
        self.weights = None
        self.bias = None

    def forward(self, input):
        """
        Perform the forward pass through the layer.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output of the layer after applying activation, shape (batch_size, output_size).
        """
        pass
    
    def backward(self, grad_output):
        """
        Perform the backward pass through the layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input, shape (batch_size, input_size).
        """
        pass
    
class FeedForwardNet:
    def __init__(
        self, input_size, output_size, optimizer, 
        hidden_layers=[8, 16, 32], 
        output_activation=SoftmaxActivation, 
        intermittent_activation=ReLUActivation
    ) -> None:
        """
        Initialize a feedforward neural network.

        Args:
            input_size (int): Number of input units.
            output_size (int): Number of output units.
            optimizer: The optimizer for updating weights and biases.
            hidden_layers (list): List of hidden layer sizes.
            output_activation: Activation function for the output layer.
            intermittent_activation: Activation function for the hidden layers.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer

        self.loss_object = CrossEntropyLoss()

        #TODO: Initialize list of layers for the network.
        self.layers = []


    def forward(self, input):
        """
        Perform the forward pass through the network.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output of the network, shape (batch_size, output_size).
        """
        pass

    def backward(self, output, target):
        """
        Perform the backward pass through the network and compute the loss.

        Args:
            output (np.ndarray): Output of the network, shape (batch_size, output_size).
            target (np.ndarray): Target values, typically one-hot encoded labels.

        Returns:
            float: Loss computed using the Cross-Entropy loss function.
        """
        pass
