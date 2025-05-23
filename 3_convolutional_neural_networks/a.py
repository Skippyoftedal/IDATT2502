import torch
import torch.nn as nn
import torchvision

from utility.deviceUtility import get_best_available_device

# Check for CUDA availability
device = get_best_available_device()

# Load observations from the MNIST dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('../data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)  # Move data to GPU
y_train = torch.zeros((mnist_train.targets.shape[0], 10), device=device)  # Create output tensor on GPU
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('../data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)  # Move data to GPU
y_test = torch.zeros((mnist_test.targets.shape[0], 10), device=device)  # Create output tensor on GPU
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalize inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(64 * 7 * 7, 10)

    def logits(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return self.dense(x.reshape(-1, 64 * 7 * 7))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


# Instantiate the model and move it to GPU
model = ConvolutionalNeuralNetworkModel().to(device)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Training loop
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        # Move each batch to GPU
        x_batch = x_train_batches[batch]
        y_batch = y_train_batches[batch]

        # Compute loss gradients and optimize
        model.loss(x_batch, y_batch).backward()
        optimizer.step()  # Perform optimization
        optimizer.zero_grad()  # Clear gradients for next step

    print(f"accuracy = {model.accuracy(x_test, y_test).item()}")

# accuracy = 0.9708999991416931
# accuracy = 0.9811999797821045