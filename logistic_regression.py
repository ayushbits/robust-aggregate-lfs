
# import torch
import torch.nn as nn
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
        # .to(device=device)