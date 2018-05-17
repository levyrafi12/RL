import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss

# TODO: define network, loss and optimiser(use learning rate of 0.1).
# Hyper Parameters
input_size = env.observation_space.n
num_classes = env.action_space.n
batch_size = 1
learning_rate = 0.1

 # Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out
 
def one_hot_state_encoding(state):
    oh_vec = torch.zeros(input_size)
    oh_vec[state] = 1
    return Variable(oh_vec)

net = Net(input_size, num_classes)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # env.render()
        # 1. Choose an action greedily from the Q-network
        # (run the network for current state and choose the action with the maxQ)
        Q = net(one_hot_state_encoding(s))
        _, a = torch.max(Q, 0)
        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()
        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        # print("a = %d" % (int(a[0])))
        s1, r, d, _ = env.step(int(a[0]))
        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        rAll += r
        Q1 = net(one_hot_state_encoding(s1))
        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        Q_target = copy.copy(Q.data)
        Q_target[int(a[0])] = r + y * torch.max(Q1.data)
        # 6. Train the network using target and predicted Q values (model.zero(), 
        # forward, backward, optim.step)
        Q = net(one_hot_state_encoding(s))
        loss = criterion(Q, Variable(Q_target))
        optimizer.zero_grad()   # zero the gradient buffers
        loss.backward()
        # print("loss = %f" % (loss.data[0]))
        optimizer.step()

        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            """
            if r > 0:
                print("j = %d , reach Goal" % (j))
            else:
                 print("j = %d , reach hole" % (j))
            """
            break

    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))