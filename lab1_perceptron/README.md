# Introduction

The subject of this laboratory will be to implement a simple perceptron for binary classification on MNIST dataset. The perceptron will be implemented in pure Python and NumPy. The goal of this laboratory is to get familiar with the basic concepts of neural networks and to understand the basic building blocks of neural networks.

# Tasks for this laboratory
Each task described below should be filled in according to the prototype classes provided within perceptron_utils.py. The prototype classes are not complete and you will need to implement the missing parts. The prototype classes are provided to help you get started with the laboratory. You are free to modify the prototype classes as you see fit.
## Task 1: Implementing tools for gradient descent
### Task 1.1: Implementing the loss function class
For the purpose of this laboratory we will use the binary cross-entropy loss function. The binary cross-entropy loss function is defined as follows:
$$\text{BCE}(y, \hat{y}) = - \left( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right)$$
Where:
- $y$ is the true label
- $\hat{y}$ is the predicted label
- $\log$ is the natural logarithm

This however is not numerically stable algorithm. To avoid numerical instability we will use the following equivalent formulation:
$$\text{BCE}(y, \hat{y}) = - \left( y \log(\hat{y} + \varepsilon) + (1 - y) \log(1 - \hat{y} + \varepsilon) \right)$$
Where:
- $\varepsilon$ is a small number, e.g. $10^{-7}$
The gradient of the binary cross-entropy loss function is defined then as:
$$\frac{\partial \text{BCE}(y, \hat{y})}{\partial \hat{y}} = -\left( \frac{y}{\hat{y} + \varepsilon} - \frac{1-y}{1-\hat{y} + \varepsilon} \right)$$
Implement the binary cross-entropy loss function and its gradient in the BinaryCrossEntropyLoss class. Both loss and gradient should be implemented within the forward method. The loss function should be implemented in a way that it can handle both single samples and batches of samples. The gradient should be averaged over the batch dimension.

### Task 1.2: Implementing the SGD class
Implement the SGD class. The SGD class should implement the simple stochastic gradient descent algorithm. The SGD class should implements the following methods:
- \_\_init\_\_ - initializes the SGD class (sets the learning rate, and clipping threshold -- already implemented)
- update_weights - updates the weights of the model
- update_bias - updates the bias of the model

The update is given by the following formula:
$$\theta \leftarrow \theta - \alpha \frac{1}{M} \sum_{i=1}^{M} \frac{\partial L}{\partial \theta}(x^{(i)}, y^{(i)})$$
Where:
- $\theta$ is the parameter to be updated (weight or bias)
- $\alpha$ is the learning rate
- $M$ is the batch size
- $L$ is the loss function
- $x^{(i)}$ is the $i$-th sample in the batch
- $y^{(i)}$ is the $i$-th label in the batch

If you are averaging the loss over the batch dimension, you do not need to sum over the batch dimension and scale in the update rule. This then becomes:
$$\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}(x, y)$$

## Task 2: Implementing a perceptron
### Task 2.1: Implementing the sigmoid activation function class
Implement the sigmoid activation function class. The sigmoid activation function is defined as follows:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
The gradient of the sigmoid activation function is defined as follows:
$$\frac{\partial \sigma(x)}{\partial x} = \sigma(x) \cdot (1 - \sigma(x))$$
Implement the sigmoid activation function and its gradient in the SigmoidActivation class. Computing the function value should be implemented in the forward method while computing the gradient should be implemented in the backward method. Both the forward and backward methods should be able to handle both single samples and batches of samples.

### Task 2.2: Implementing the perceptron class
Implement the perceptron class. The perceptron class should implement a simple perceptron with a sigmoid activation function. The perceptron class should implement the following methods:
- \_\_init\_\_ - initializes the perceptron class (already implemented), and initializes the weights and bias (not implemented)
- forward - computes the forward pass of the perceptron
- backward - computes the backward pass of the perceptron

The forward pass of the perceptron is defined as follows:
$$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)$$
Where:
- $\hat{y}$ is the predicted label
- $\sigma$ is the sigmoid activation function
- $\mathbf{w}$ is the weight vector
- $\mathbf{x}$ is the input vector
- $b$ is the bias

During the backward pass the following gradients should be computed:
- Gradient of the activation function, given by sigmoid.backward()
- Weight gradient, here is given by:
$$\nabla \theta = x^T \cdot \delta$$
Where:
    - $\theta$ is the weight vector
    - $x$ is the input vector
    - $\delta$ is the gradient of the loss function with respect to the output of the perceptron (sigmoid.backward() in this case)
- Bias gradient, here is given by:
$$\nabla b = \delta$$
Where:
- $b$ is the bias
- $\delta$ is the gradient of the loss function with respect to the output of the perceptron (sigmoid.backward() in this case)

After computing the gradients, the weights and bias should be updated using the SGD class.

## Task 3: Training a perceptron
Look at the provided code within the training_loop.py. 
### Task 3.1: Implementing the training loop
Implement the training loop. The training loop should implement the following steps:
- Compute the forward pass of the perceptron
- Compute the loss and its gradient
- Compute the backward pass of the perceptron and update the weights and bias
- Compute the accuracy of the perceptron using "compute_accuracy" function
- return the loss and accuracy

### Task 3.2: Evaluating the perceptron
Implement the compute_accuracy function. The compute_accuracy function should compute the accuracy of the perceptron on the given dataset.
