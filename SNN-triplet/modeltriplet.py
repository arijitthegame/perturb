import torch
import torch.nn as nn


class Siamese(nn.Module):

    def __init__(self,input_size = 100, hidden_size1=300, hidden_size2=200, output_size = 1):

        super(Siamese, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
        )
        self.liner = nn.Sequential(nn.Linear(hidden_size1, output_size), nn.Sigmoid())
        # self.out = nn.Linear(output_size, 1)

    def forward_one(self, x):
        x = self.mlp(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x


    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1,out2


class Triplet(nn.Module):

    def __init__(self,input_size = 100, hidden_size1=300, hidden_size2=200, output_size = 1):

        super(Triplet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
        )
        self.liner = nn.Sequential(nn.Linear(hidden_size1, output_size), nn.Sigmoid())
        # self.out = nn.Linear(output_size, 1)

    def forward_one(self, x):
        x = self.mlp(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x


    def forward(self, x1, x2, x3):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out3 = self.forward_one(x3)
        return out1,out2,out3

# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
