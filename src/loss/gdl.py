import torch
import torch.nn as nn


class GDL(nn.Module):

    def __init__(self):
        super(GDL, self).__init__()
        self.alpha = 2

    def forward(self, Y_true, Y_pred):
        
        print(Y_true.shape)
        print(Y_pred.shape)
        t1 = torch.pow(torch.abs(Y_true[:, :, :, :] - Y_true[:, :, ::-1, :]) -
                   torch.abs(Y_pred[:, :, :, :] - Y_pred[:, :, ::-1, :]), self.alpha)
        t2 = torch.pow(torch.abs(Y_true[:, :, :, :-1] - Y_true[:, :, :, :]) -
                   torch.abs(Y_pred[:, :, :, ::-1] - Y_pred[:, :, :, :]), self.alpha)
        
        print(t1.shape)
        print(t2.shape)

        loss = torch.mean((t1 + t2).reshape(-1), -1)

        error = torch.add(t1, t2)
        loss = torch.sum(error)
        return loss 

