# Pytorch models
## Creating a custom model

Constructing a custom neural network in PyTorch involves defining a model class that inherits from torch.nn.Module. The provided example, CustomModel, showcases a simple architecture with one hidden layer. The ```__init__``` method initializes the layers, including a linear layer and a ReLU activation function. The ```forward``` method specifies the forward pass of the model. It takes an input tensor x, applies the layers sequentially, and returns the output tensor.

```
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

## Using and exploring the model
**Important: torchinfo is an external library that can be installed with pip ([more info](https://github.com/TylerYep/torchinfo))**

Exploring a created model can be done easily through tensorboard graphs or torchinfo summary which example is provided below.
```
from torchinfo import summary

input_size = 10
hidden_size = 20
output_size = 5
batch_size = 3

model = CustomModel(input_size, hidden_size, output_size)
summary(model, (batch_size, input_size))
```
Utilizing the custom model for predictions begins with preparing input data. In this case, a random input tensor is generated for demonstration purposes. The ```torch.no_grad()``` context ensures that no gradients are calculated during inference, as we are not training the model.
```
sample_input = torch.randn((1, input_size))
with torch.no_grad():
    predictions = model(sample_input)
print("Predictions:", predictions)
```
# Training the models
## Loss functions and validation metrics
Most of the loss functions are already [implemented within pytorch](https://pytorch.org/docs/stable/nn.html#loss-functions). If you would like to create a custom loss function, its very simple.
```
class loss_function(nn.Module):
    def __init__(self, weight_mse=1.0, weight_bce=1.0):
        super(HybridLoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_bce = weight_bce

    def forward(self, output_continuous, target_continuous, output_binary, target_binary):
        # Calculate Mean Squared Error (MSE) loss for continuous outputs
        mse_loss = F.mse_loss(output_continuous, target_continuous)

        # Calculate Binary Cross Entropy (BCE) loss for binary outputs
        bce_loss = F.binary_cross_entropy_with_logits(output_binary, target_binary)

        # Combine the two losses with specified weights
        hybrid_loss = self.weight_mse * mse_loss + self.weight_bce * bce_loss

        return hybrid_loss
```
## Simple training loop
```
num_epochs = 10
# Initialize Stochastic Gradient Descent optimizer with 
# learning rate and hooks for all the model parameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Iterate over next epochs 
# (whole training dataset seen by the model)
for epoch in range(num_epochs):
    # Iterate over batches of data
    for batch_data in dataloader:
        inputs, labels = batch_data

        # Forward pass
        predictions = model(inputs)
        # Compute loss
        loss = loss_function(predictions, labels)
        # Backward pass and optimization
        # Zero any previous gradients that were computed to start with blank slate
        optimizer.zero_grad()
        # Compute new gradients based on the loss function
        loss.backward()
        # Make a step optimizer, updating all parameters that were 
        # provided to it, based on computed gradients
        optimizer.step()

    # Validation and metrics evaluation
    # ...
```
## Further reading:
- [Torch beginner tutorial](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- Previous laboratory links
- [Pytorch examples](https://github.com/pytorch/examples) especially [MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py)
- [Stanford cource](https://cs230.stanford.edu/blog/pytorch/)
## Tasks:

1. Create a simple general MLP model. During initialisation it should recieve an activation function as well as each layer size (depending on the list, the depth of the model might be different). And use the model on a sample input to show it's shape after the output.
2. Use the created model to train a simple MNIST recognition model (dataset can be imported [from torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)). Write a training loop that collects losses and accuracies. You can repurpose the code from previous laboratories for that.
3. Change the dataset to your own dataset written on previous laboratory. What do you see?