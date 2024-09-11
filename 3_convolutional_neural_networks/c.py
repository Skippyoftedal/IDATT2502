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

        #Filtrerer etter forskjellige egenskaper?
        #Blir initialisert med random verdier, som gjør at filtrene ikke blir like
        #out_channels betyr antall filtre
        #kernel size 5 => 5x5 filter
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        #Vanligvis ligger ReLU rett etter convolutional lagene
        self.relu = nn.ReLU()
        #Reduserer antall piksler, slik at man får med det viktigste?
        self.pool1 = nn.MaxPool2d(kernel_size=2)



        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(64* 7*7, 1024)
        self.dense2 = nn.Linear(1024, 10)

        #Dropout-lag bør antakeligvis ligge på slutten, den originale artikkelen foreslår 0.5
        #Man kan også vurdere rundt 0.1 eller 0.2 for for conv lagene?
        #https://stats.stackexchange.com/a/317313
        self.dropout = nn.Dropout(0.5)

    def logits(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)#Må alltid flates ut før dense?
        x = self.dense(x)
        x = self.dropout(x)
        return self.dense2(x)

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), 0.0001)


accuracies = []
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        x_batch = x_train_batches[batch]
        y_batch = y_train_batches[batch]

        optimizer.zero_grad()
        model.loss(x_batch, y_batch).backward()
        optimizer.step()
    ac = model.accuracy(x_test, y_test).item()
    accuracies.append(ac)
    print(f"accuracy = {ac}")

maxVal = max(accuracies)
print(f"The max accuracy is {maxVal}")

"""
Diverse forsøk
1: lik som oppgave b: 0.983199954032898
2: dropout (p=0.5): 0.9845999479293823
3: 2 stk relu + dropout: 0.9932999610900879
"""