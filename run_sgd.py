import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model import convnet
from functools import partial
# from Dx_losses import Dx_cross_entropy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conv_hidden_size = (6, 16)
dense_hidden_size = (120, 84)

Net = partial(convnet.LeNet5, 10, 3, conv_hidden_size, dense_hidden_size, device)


transform = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
     ])

trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./datasets/cifar/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = Net().to(device)
criterion = nn.CrossEntropyLoss()

net.requires_grad_(True)

# def Dx_cross_entropy(input, one_hot):
#     return torch.softmax(input, dim=1) - one_hot
#
# def criterion(f_data, label):
#     return torch.mean(torch.sum(Dx_cross_entropy(f_data, label) ** 2, dim=(1,)))

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(1000):  # loop over the dataset multiple times
    # running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # loss = criterion(outputs, labels_onehot)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 200 == 199:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #       (epoch + 1, i + 1, running_loss / 2000))
            # running_loss = 0.0

    correct = 0
    loss = 0
    total = 0
    # with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # loss += criterion(outputs, labels_onehot).item()

    print(f'Accuracy of the network on the 10000 test images: {100. * correct / total} %%')
    # print(f'Loss of the network on the 10000 test images: {loss} %%')
