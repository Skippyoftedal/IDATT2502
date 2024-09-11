import torch
import torch.nn as nn
import torchvision
from functorch.dim import Tensor

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA is available, using GPU')
else:
    device = torch.device("cpu")
    print('CUDA is not available, using CPU')

mnist_train = torchvision.datasets.MNIST('../data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)  # Move data to GPU
y_train = torch.zeros((mnist_train.targets.shape[0], 10), device=device)  # Create output tensor on GPU
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('../data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)  # Move data to GPU
y_test = torch.zeros((mnist_test.targets.shape[0], 10), device=device)  # Create output tensor on GPU
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

BATCHES = 600
x_train_batches: tuple[Tensor, ...] = torch.split(x_train, BATCHES)
y_train_batches: tuple[Tensor, ...] = torch.split(y_train, BATCHES)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(64* 7*7, 1024)
        self.dense2 = nn.Linear(1024, 10)

    def logits(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the dense layer
        x = self.dense(x)
        return self.dense2(x)

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), 0.001)

for epoch in range(20):
    for batch in range(len(x_train_batches)):
        x_batch = x_train_batches[batch]
        y_batch = y_train_batches[batch]

        model.loss(x_batch, y_batch).backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"accuracy = {model.accuracy(x_test, y_test).item()}")
