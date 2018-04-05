import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import mnist_stat as stat

def basic_config():
    # Hyper Parameters
    input_size = 784
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
        def __init__(self, input_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            return out


    net = Net(input_size, num_classes)


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    running_loss = 0
    epoch_hist = []
    loss_hist = []

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
            if i % 200 == 199: # print statistics
                epoch_hist, loss_hist = stat.print_statistics(
                    epoch_hist, loss_hist, epoch, i, running_loss)        
                running_loss = 0.0
            optimizer.step() # update the weights

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

    # stat.plot_curve(epoch_hist, loss_hist, 'minst_loss_curve_1.png')

    # Save the Model
    # torch.save(net.state_dict(), 'model.pkl')
    return epoch_hist, loss_hist

if __name__ == "__main__":
    basic_config()