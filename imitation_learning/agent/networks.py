import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):
  def __init__(self, history_length=0, n_classes=3):
    super(CNN, self).__init__()
    #   define layers of a convolutional neural network
    #   self.cnn_layers = nn.Sequential(
    #   nn.Conv2d(history_length + 1, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    #   nn.ReLU(inplace=True),
    #   nn.MaxPool2d(kernel_size=2, stride=2),
    #   # Another Conv layer2(96-2-2 /2 = 48)
    #   nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    #   nn.ReLU(inplace=True),
    #   nn.MaxPool2d(kernel_size=2, stride=2),
    #   # Another Conv layer3 (46-2-2 /2 = 24)
    #   nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    #   nn.ReLU(inplace=True),
    #   nn.MaxPool2d(kernel_size=2, stride=2),
    #   # op 24\2=12
    # )
    # self.linear_layers = nn.Sequential(
    #   nn.Linear(12 * 12 * 48, 25),
    #   nn.ReLU(),
    #   nn.Linear(25, 5)
    # )
    self.conv1 = nn.Conv2d(history_length + 1, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv2 = nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv3 = nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.fc1 = nn.Linear(12*12*48, 25)
    self.fc2 = nn.Linear(25, 5)
    self.maxp = nn.MaxPool2d(2, 2)

  def forward(self, x):
    # compute forward pass
    # x=x.int()
    x = torch.tensor(x).cuda()
    # x = self.cnn_layers(x)
    # x = torch.flatten(x, start_dim=1)
    # x = self.linear_layers(x)
    x = self.conv1(x)
    x = F.relu(x)
    x = self.maxp(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.maxp(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.maxp(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

