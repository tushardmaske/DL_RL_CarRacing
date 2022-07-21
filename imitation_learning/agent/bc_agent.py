import torch
from imitation_learning.agent.networks import CNN

class BCAgent:
    
    def __init__(self,history_length=0):
        # Define network, loss function, optimizer
        self.net = CNN(history_length).cuda()
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

    def update(self, X_batch, y_batch):
        # transform input to tensors
        # forward + backward + optimize
        X_batch1 = torch.tensor(X_batch).cuda()
        y_batch1 = torch.tensor(y_batch).cuda()

        self.optimizer.zero_grad()
        #X_batch1 = torch.unsqueeze(X_batch1, dim=1)
        y_train = self.net(X_batch1)
        loss = self.loss(y_train, y_batch1)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, X):
        # forward pass
        X = torch.tensor(X).float().cuda()
       # X = torch.unsqueeze(X, dim=1)
        outputs = self.net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
