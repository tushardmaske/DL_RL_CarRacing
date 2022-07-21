import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

class CNN(nn.Module):
  def __init__(self, history_length=0, n_classes=3):
    super(CNN, self).__init__()
    # define layers of a convolutional neural network
    self.cnn_layers = nn.Sequential(
      nn.Conv2d(in_channels=history_length + 1, out_channels=24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # Another Conv layer2(96-2-2 /2 = 48)
      nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # Another Conv layer3 (46-2-2 /2 = 24)
      nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # op 24\2=12
    )
    self.linear_layers = nn.Sequential(
      nn.Linear(12 * 12 * 48, 25),
      nn.ReLU(),
      nn.Linear(25, 5)
    )

  def forward(self, x):
    # compute forward pass
    # x=x.int()
    x = torch.tensor(x).cuda()
    x = self.cnn_layers(x)
    x = torch.flatten(x, start_dim=1)
    x = self.linear_layers(x)
    return x


