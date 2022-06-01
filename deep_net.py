import torch.nn as nn
import torch
import torch.nn.functional as F

class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed = 7):
        super(DeepNet, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        torch.manual_seed(seed)

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = F.relu(self.linear_2(out))
        return self.out(out)