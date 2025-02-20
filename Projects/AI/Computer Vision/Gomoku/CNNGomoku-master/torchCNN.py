#665 project

import torch
import torch.nn as nn
from torch import optim
import numpy as np
cuda = torch.device('cuda')

from SGFFileProcess import SGFflie
import torch.nn.functional as F


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)


        #  policy layers
        self.act_conv1 = nn.Conv2d(32, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*15*15, 15*15)
        #  value layers
        self.val_conv1 = nn.Conv2d(32, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*15*15, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, input):
    # relu all the way
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))


        #  policy layers with softmax
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*15*15)
        x_act = F.log_softmax(self.act_fc1(x_act))

        # value layers with tanh
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*15*15)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNet():
    """policy-value network """
    def __init__(self ,gpu = True):
        if gpu is True:
            self.policy_value_net = Net().float().cuda()
        else:
            self.policy_value_net = Net().float()
        self.l2_const = 1e-4  # copied from others.
        # the policy value net module

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.policy_value_net.parameters(), lr=0.001, momentum=0.9)

    def train(self, putin,label):
        # get the inputs
        input, labels = putin, label

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        poutput, voutput = self.policy_value_net.forward(input)
        loss = self.criterion(poutput, labels)

        loss.backward()
        self.optimizer.step()
        return

    def classify(self, putin):

        input = putin
        poutput, voutput = self.policy_value_net.forward(input)
        predict = (torch.argmax(poutput).item())

        # return argmax position in 1d tensor.
        return predict