import torch.nn as nn
import torch
class LogisticReg(nn.Module):
    def __init__(self, input_size, output_size, seed = 7):
        super(LogisticReg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        torch.manual_seed(seed)

    def forward(self, x):
        return self.linear(x)
