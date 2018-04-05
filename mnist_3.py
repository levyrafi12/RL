import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import time
import mnist_stat as stat

def hidden_layer_config():
    # Hyper Parameters
    input_size = 784
    hidden_layer_size = 500
    num_classes = 10
    num_epochs = 100
    batch_size = 100
    learning_rate = 1e-3

    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

    test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

    # Neural Network Model
    class Net(nn.Module):
        def __init__(self, input_size, hidden_layer_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_layer_size)
            self.fc2 = nn.Linear(hidden_layer_size, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            out = self.fc2(x)
            return out


    net = Net(input_size, hidden_layer_size, num_classes)


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)

    running_loss = 0

    loss_hist = []
    epoch_hist = []

    # in your training loop:

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            # TODO: implement training code
            output = net(images) # Forward
            loss = criterion(output, labels)
            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            running_loss += loss.data[0]
            # print statistics
            if i % 200 == 199:
                epoch_hist, loss_hist = stat.print_statistics(
                    epoch_hist, loss_hist, epoch, i, running_loss)
                running_loss = 0.0
            optimizer.step()    

    # stat.plot_curve(epoch_hist, loss_hist, 'mnist_loss_curve_3.png')

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # TODO: implement evaluation code - report accuracy
        output = net(images) # Forward
        _, predicted = torch.max(output, 1)
        for j in range(batch_size):
            if (predicted.data[j] == labels.data[j]):
                correct += 1
        total += labels.size(0)

    print('Accuracy of the network on the 10000 test images: %d%%' % (100 * correct / total))

    # stat.plot_curve(epoch_hist, loss_hist, 'minst_loss_curve_3.png')
    return epoch_hist, loss_hist

    # Save the Model
    # torch.save(net.state_dict(), 'model.pkl')

if __name__ == "__main__":
    hidden_layer_config()