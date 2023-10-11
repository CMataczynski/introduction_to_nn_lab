import numpy as np

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
        # ReLU function: f(x) = max(0, x)
        return np.maximum(input, 0)

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of ReLU activation.

        Args:
            input (np.ndarray): Input data.
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        # The derivative of ReLU:
        # f'(x) = 1 if x > 0, 0 otherwise
        relu_grad = input > 0
        return grad_output * relu_grad  

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
        self.sigmoid = 1 / (1 + np.exp(-input))
        return self.sigmoid

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
        return grad_output * self.sigmoid * (1 - self.sigmoid)

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
        
        # Softmax function for each row:
        # f(x)_i = exp(x_i) / sum(exp(x_j) for j in range(num_classes))
        return exponents / np.sum(exponents, axis=1, keepdims=True)

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of Softmax activation.

        Args:
            input (np.ndarray): Input data of shape (batch_size, num_classes).
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input, with the same shape as input.
        """
        softmax = self.forward(input)
        
        # The derivative of the Softmax function:
        # f'(x)_i = f(x)_i * (1 - f(x)_i) for i
        # f'(x)_i = -f(x)_i * f(x)_j for j != i
        return grad_output * softmax * (1 - softmax)


import numpy as np

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
        # Clip predictions to prevent numerical instability
        predictions = np.clip(predictions, self.stability, 1 - self.stability)
        
        # Compute the loss using the Cross-Entropy formula with stability
        loss = -np.sum(target * np.log(predictions + self.stability) + (1 - target) * np.log(1 - predictions + self.stability)) / predictions.shape[0]
        
        # Compute the gradient of the loss with respect to predictions
        grad = (predictions - target) / (predictions * (1 - predictions) + self.stability) / predictions.shape[0]
        
        return loss, grad
    
    def empty_grad(self):
        """
        Create an empty gradient array of the same shape as the gradient computed during forward pass.

        Returns:
            np.ndarray: An array of zeros with the same shape as the gradient.
        """
        return np.zeros_like(self.grad)

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

class Perceptron:
    def __init__(self, input_size, activation, optimizer) -> None:
        """
        Initialize a fully connected layer.

        Args:
            input_size (int): Number of input units.
            activation: The activation function to use (e.g., sigmoid, ReLU).
            optimizer: The optimizer for updating weights and biases (e.g., SGD, Adam).
        """
        self.input_size = input_size
        self.activation = activation
        self.optimizer = optimizer  

        self.weights = np.random.randn(input_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros(1)
    
    def forward(self, input): 
        """
        Perform the forward pass through the layer.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output of the layer after applying activation, shape (batch_size, 1).
        """
        self.input = input
        perceptron = np.dot(input, self.weights) + self.bias
        self.activation_input = perceptron
        output = self.activation.forward(perceptron)
        return output
    
    def backward(self, grad_output):
        """
        Perform the backward pass through the layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input, shape (batch_size, input_size).
        """
        grad_input = self.activation.backward(self.activation_input, grad_output)
        self.weights_grad = np.dot(self.input.T, grad_input)
        self.bias_grad = np.sum(grad_input, axis=0)
        self.weights = self.optimizer.update_weights(self.weights, self.weights_grad)
        self.bias = self.optimizer.update_bias(self.bias, self.bias_grad)
        return np.dot(grad_input, self.weights.T)


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
        
        # Kaiming He initialization for weights
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros(output_size)
        self.empty_grad()

    def empty_grad(self):
        """
        Initialize gradient arrays for weights and biases with zeros.
        """
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, input):
        """
        Perform the forward pass through the layer.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output of the layer after applying activation, shape (batch_size, output_size).
        """
        self.input = input
        perceptron = np.dot(input, self.weights) + self.bias
        self.activation_input = perceptron

        # Compute the weighted sum (perceptron) of inputs:
        # \[ \text{perceptron} = \mathbf{X} \cdot \mathbf{W} + \mathbf{b} \]
        
        output = self.activation.forward(perceptron)

        # Apply the activation function to the perceptron:
        # \[ \text{output} = \sigma(\text{perceptron}) \]

        return output
    
    def backward(self, grad_output):
        """
        Perform the backward pass through the layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input, shape (batch_size, input_size).
        """
        grad_input = self.activation.backward(self.activation_input, grad_output)

        # Compute the gradient of the loss with respect to the input:
        # \[ \text{grad\_input} = \frac{{d\text{Loss}}}{{d\text{perceptron}}} \cdot \frac{{d\text{perceptron}}}{{d\text{input}}} \]

        self.weights_grad = np.dot(self.input.T, grad_input)
        
        # Compute the gradient of the loss with respect to the weights:
        # \[ \text{weights\_grad} = \frac{{d\text{Loss}}}{{d\text{perceptron}}} \cdot \frac{{d\text{perceptron}}}{{d\text{weights}}} \]

        self.bias_grad = np.sum(grad_input, axis=0)

        # Update the weights using the optimizer:
        # \[ \text{weights} = \text{weights} - \text{learning\_rate} \cdot \text{weights\_grad} \]
        
        self.weights = self.optimizer.update_weights(self.weights, self.weights_grad)

        # Update the bias using the optimizer:
        # \[ \text{bias} = \text{bias} - \text{learning\_rate} \cdot \text{bias\_grad} \]

        self.bias = self.optimizer.update_bias(self.bias, self.bias_grad)

        return np.dot(grad_input, self.weights.T)
    
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

        # Create layers for the network
        self.layers = [
            FullyConnectedLayer(input_size, hidden_layers[0], intermittent_activation(), optimizer())
        ]
        for i in range(len(hidden_layers) - 1):
            self.layers.append(
                FullyConnectedLayer(hidden_layers[i], hidden_layers[i + 1], intermittent_activation(), optimizer())
            )
        self.layers.append(
            FullyConnectedLayer(hidden_layers[-1], output_size, output_activation(), optimizer())
        )

    def forward(self, input):
        """
        Perform the forward pass through the network.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output of the network, shape (batch_size, output_size).
        """
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output, target):
        """
        Perform the backward pass through the network and compute the loss.

        Args:
            output (np.ndarray): Output of the network, shape (batch_size, output_size).
            target (np.ndarray): Target values, typically one-hot encoded labels.

        Returns:
            float: Loss computed using the Cross-Entropy loss function.
        """
        loss, grad = self.loss_object.forward(output, target)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss
