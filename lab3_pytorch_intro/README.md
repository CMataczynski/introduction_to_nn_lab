# Pytorch

## Overview

We will now dive into modern tools for building neural networks in Python, those include, but are not limited to:
- [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) -- Google's in house created package to run highly optimized and highly distributed computation on CPUs, GPUs and TPUs
- [Pytroch](https://pytorch.org/get-started/locally/) -- open code software with easy pythonic interface and support for CPU and GPU training (with TPUs added in newer versions)
- [Tensorflow](https://www.tensorflow.org/install?hl=en) -- another google project with less intuitive and boilerplate-heavy code (at least before tensorflow 2.0. I have not used it since.)
- Other-than-Python languages and their implementations of deep learning software.

For this laboratory we will use Pytorch. Please install it according to your PC specs and the quickstart linked above.

## Data flow in pytorch

### Tensors, variables, types and devices

In very high overview, Pytorch dataflow consists of two types: Tensors and Parameters. Both are very similar to each other with one key difference. Tensors are a simple multidimensional data storage, that has an arbitrary shape, set data type (dtype) and device on which it's stored.

A simple tensor can be created with:
```
tensor = torch.Tensor([[0,1,2],[1,0.2,0.3]])
# tensor([[0.0000, 1.0000, 2.0000],
#        [1.0000, 0.2000, 0.3000]])

tensor.dtype
# torch.float32

tensor.shape
# torch.Size([2, 3])

tensor.device
# device(type='cpu')

tensor.requires_grad
# False
```
similar Parameter can be created by:
```
parameter = torch.nn.Parameter(tensor)
# Parameter containing:
# tensor([[0.0000, 1.0000, 2.0000],
#         [1.0000, 0.2000, 0.3000]], requires_grad=True)
```
The difference is visible even here. A Parameter is used two-fold in Pytorch. Firstly as a simple way to let the engine know that this set of parameters will need gradient stored for it's operations, secondly it's a way for Pytorch to know how many learnable parameters there are in a whole model. If it's needed, any tensor can compute it's gradients by specifying requires_grad parameter as true.

### Tensor operations

Let's assume three tensors:
```
tensor23 = torch.Tensor([[0,1,2],[1,0.2,0.3]])
# torch.Size([2, 3])

tensor31 = torch.Tensor([[0], [1], [0]])
# torch.Size([3, 1])

tensor21 = torch.Tensor([[1], [2]])
# torch.Size([2, 1])
```

If the shapes of those tensors can be broadcasted, they will be elligible to be used in different operations. Simple ones are:
- Addition/Substraction/Element-wise multiplication/division:
```
tensor23 + tensor21
# tensor([[1.0000, 2.0000, 3.0000],
#         [3.0000, 2.2000, 2.3000]])

tensor23 - tensor21
# tensor([[-1.0000,  0.0000,  1.0000],
#         [-1.0000, -1.8000, -1.7000]])

tensor23 * tensor21
# tensor([[0.0000, 1.0000, 2.0000],
#         [2.0000, 0.4000, 0.6000]])

tensor23 / tensor21
# tensor([[0.0000, 1.0000, 2.0000],
#         [0.5000, 0.1000, 0.1500]])
```
- Mathematical multiplication
```
torch.matmul(tensor23, tensor31)
# tensor([[1.0000],
#         [0.2000]])
```
- Reshaping
```
tensor61 = tensor23.reshape((6,1))
# tensor([[0.0000],
#         [1.0000],
#         [2.0000],
#         [1.0000],
#         [0.2000],
#         [0.3000]])

tensor61.shape
# torch.Size([6, 1])
```

The caveat is that the tensors must be of the same (or compatible) types, and be on the exact same device to be elligible for those operations. Both can be changed with operator ```to```:
```
tensor21.to(torch.float16)
# tensor([[1.],
#         [2.]], dtype=torch.float16)
tensor21.to(device=torch.device("cuda"))
# tensor([[1.],
#         [2.]], device='cuda:0')
```

## Datasets and Dataloaders
There are few simple dataset forms already implemented available in:

| Problem | Pre-built Dataset location |
| ------- | -------------------------- |
| Vision | torchvision.datasets |
| Audio | torchaudio.datasets |
| Text | torchtext.datasets |
| Recommender systems | torchrec.datasets |

But most of the time we will be implementing own, custom datasets available through:
- [Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
    - It's a basic class that implements pipeline for loading the data, getting data by providing index and finding how much of the data is available
    - It needs to implement ```__init__```, ```__len__```,```__getitem__```
- [DataLoaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    - It's used to easily and optimally iterate over Dataset, batching it with many workers.
    - Needs Dataset object for initialization.

Basic usage can be seen [here.](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)


# Materials to read/use:
 - [Official begginer guide](https://pytorch.org/tutorials/beginner/basics/intro.html)
 - [Official torch tutorials](https://pytorch.org/tutorials/)
 - [Medium blogs and posts](https://towardsdatascience.com/pytorch-for-deep-learning-a-quick-guide-for-starters-5b60d2dbb564)
 - [Learn pytorch.io](https://www.learnpytorch.io/00_pytorch_fundamentals/)
 - [Official documentation](https://pytorch.org/docs/stable/index.html)


# Tasks:
1. Define your M, N that will be the number of inputs and outputs to the perceptron layer of NN and using only torch:
    - Draw weights of shape (M, N) from normal distribution with mean 0 and std 1
    - Create a random input vector of shape (3, M)
    - Compute an output of the layer with [Sigmoid activation function](https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html#torch.nn.functional.sigmoid)
    - Print shape of the outputted tensor
2. Implement a simple Dataset loader for images of cats and dogs. Download and use your own images. Store them in a single directory with any structure that you'd like.
    - The Dataset should recieve path to that directory as an input.
    - It should find all the images with provided extension
    - It should store internally all the labels
    - It should load the images during ```__getitem__``` not during ```__init__```
    - Make sure that after loading all the data is reshaped into a shape provided by the user in the ```__init__``` method
    - Use the dataset to load a few images and print them with respected labels using ```matplotlib```

3. Implement a simple dataloader for created dataset. Test it's speed when using different number of workers and different batch size.
    - Test whether the data is loaded asynchronously (can be done using prints)