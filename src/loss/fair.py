import torch
import torch.nn as nn

class Fair(nn.Module):

    def __init__(self):
        super(Fair, self).__init__()
        self.c = 1

    def forward(self, X, Y):
        r = torch.add(X, -Y)
        r_a = torch.abs(r)

        loss = self.c * self.c * ( r_a/self.c - torch.log(1 + r_a/self.c ) )
        
        return loss 
