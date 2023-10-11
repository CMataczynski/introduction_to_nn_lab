# Introduction

This laboratory will walk you through creating your first Fully connected neural network and training it on MNIST dataset.


# Tasks for this laboratory
Each task described below should be filled in according to the prototype classes provided within mlp_utils.py. The prototype classes are not complete and you will need to implement the missing parts. The prototype classes are provided to help you get started with the laboratory. You are free to modify the prototype classes as you see fit. You will only need to modify the code within the mlp_utils.py file.

## Task 1: Implementing activation functions
### Task 1.1: Implementing the sigmoid activation function class
You can take this code from the previous laboratory.

### Task 1.2: Implementing the ReLU activation function class
Implement the ReLU activation function class. The ReLU activation function is defined as follows:
$$\text{ReLU}(x) = \max(0, x)$$
The gradient of the ReLU activation function is defined as follows:
$$\frac{\partial \text{ReLU}(x)}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$
Implement the ReLU activation function and its gradient in the ReLUActivation class. Computing the function value should be implemented in the forward method while computing the gradient should be implemented in the backward method. Both the forward and backward methods should be able to handle both single samples and batches of samples.

### Task 1.3: Implementing the softmax activation function class
Implement the softmax activation function class. The softmax activation function is defined as follows:
$$\text{softmax}(\mathbf{x}) = \frac{e^{\mathbf{x}}}{\sum_{i=1}^{N} e^{x_i}}$$
Where:
- $\mathbf{x}$ is a vector of values
- $N$ is the number of elements in the vector

It's gradient is defined as follows:
$$\frac{\partial \text{softmax}(\mathbf{x})}{\partial x_i} = \text{softmax}(\mathbf{x}) \cdot (1 - \text{softmax}(\mathbf{x}))$$

Implement the softmax activation function and its gradient in the SoftmaxActivation class. Computing the function value should be implemented in the forward method while computing the gradient should be implemented in the backward method. Both the forward and backward methods should be able to handle both single samples and batches of samples.

### Task 1.4: Implemening the CrossEntropyLoss class
Implement the cross-entropy loss function class. The cross-entropy loss function is defined as follows:
$$\text{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^{N} y_i \log(\hat{y}_i)$$
Where:
- $\mathbf{y}$ is the true label
- $\hat{\mathbf{y}}$ is the predicted label
- $N$ is the number of elements in the vector

This however is not numerically stable algorithm. To avoid numerical instability we will use the following equivalent formulation:
$$\text{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^{N} y_i \log(\hat{y}_i + \varepsilon)$$
Where:
- $\varepsilon$ is a small number, e.g. $10^{-7}$
The gradient of the cross-entropy loss function is defined then as:
$$\frac{\partial \text{CE}(\mathbf{y}, \hat{\mathbf{y}})}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i + \varepsilon}$$
Implement the cross-entropy loss function and its gradient in the CrossEntropyLoss class. Both loss and gradient should be implemented within the forward method. The loss function should be implemented in a way that it can handle both single samples and batches of samples. The gradient should be averaged over the batch dimension.

## Task 2: Implementing a fully connected neural network
### Task 2.1: Implementing the FullyConnectedLayer class
Implement the fully connected layer class. The fully connected layer class should implement a fully connected layer with a given activation function. The fully connected layer class should implement the following methods:
- \_\_init\_\_ - initializes the fully connected layer class (already implemented), and initializes the weights and bias (not implemented)
Use Kaiming He initialization for the weights and set the bias to zero.
Kaiming He initialization is defined as follows:
$$\mathbf{w} \sim \mathcal{N}(0, \sqrt{\frac{2}{\text{x}_{\text{dim}}}})$$
Where:
- $\mathbf{w}$ is the weight vector
- $\text{x}_{\text{dim}}$ is the number of input units
- $\mathcal{N}$ is the normal distribution
- $\sim$ means "is sampled from"

Forward and backward passes of this layer are analogous to the forward and backward passes of the perceptron from the previous laboratory. Use them as a reference when implementing the forward and backward methods.


### Task 2.2: Implementing the FullyConnectedNetwork class
Implement the fully connected network class. The fully connected network class should implement a fully connected network with a given number of layers and a given activation function. The fully connected network class should implement the following methods:
- \_\_init\_\_ - initializes the fully connected network class (already implemented), and initializes the layers (not implemented)
- forward - computes the forward pass of the fully connected network
- backward - computes the backward pass of the fully connected network, remember to compute the loss using the loss provided within the init. 

## Task 3: Training the fully connected neural network
Use the code provided in the training_loop.py file to train the fully connected neural network. The training loop is analogous to the training loop from the previous laboratory.

### Task 3.1: Change different parameters of the network
Try changing different parameters of the network, e.g.:
- number of layers
- number of units in each layer
- activation function
- learning rate
- number of epochs
- batch size

You can easily change them within the main function of the training_loop.py file.

Do you notice any differences in the training process? Do you notice any differences in the final results?



## Final notes
This laboratory will be graded along with the next one where you will implement a simple neural network. The work you do in this laboratory will be used in the next one. You need to include following files from this laboratory:
- mlp_utils.py
- screenshots of the results of running the training_loop.py code
- short text file with remarks from the task 3.1