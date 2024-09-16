import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class NeRF(nn.Module):
    def __init__(self, filter_size=128, num_encode=6):
        super().__init__()
        self.layer1 = nn.Linear(3+3*2*num_encode, filter_size)
        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, 4)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        
        return x