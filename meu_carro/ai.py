# import modules

import numpy as np
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# creating the neural net class
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__
        self.input_size = input_size
        self.nb_action  = nb_action

        # 5 neurons in -> 30 hidden layers -> 3 out -- full connection (dense)
        self.fc1 = nn.Linear(input_size, 30) # in layer -> hidden layer
        self.fc2 = nn.Linear(30, nb_action) # hidden layer -> output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# experience replay implementation
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    # C, D, E, F, G
    # 4 values: last state, new state, last action, last reward
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0] 
    
    def sample(self, batch_size):
        # ((1, 2, 3), (4, 5, 6)) -> ((1, 4), (2, 5), (3, 6))
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
# Deep Q Learning implementation
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        # RMSProp
        self.optimizer = optim.Adam(self.model.parameter(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 7)
        








