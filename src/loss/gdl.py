import torch
import torch.nn as nn


class GDL(nn.Module):

    def __init__(self):
        super(GDL, self).__init__()
        self.alpha = 2

    def forward(self, Y_true, Y_pred):
        

        t1 = torch.pow(torch.abs(Y_true[1:, :, :] - Y_true[:-1, :, :]) -
                   torch.abs(Y_pred[1:, :, :] - Y_pred[:-1, :, :]), self.alpha)
        t2 = torch.pow(torch.abs(Y_true[:, :-1, :] - Y_true[:, 1:, :]) -
                   torch.abs(Y_pred[:, :-1, :] - Y_pred[:, 1:, :]), self.alpha)

        error = torch.add(t1, t2)
        loss = torch.sum(error)
        return loss 

