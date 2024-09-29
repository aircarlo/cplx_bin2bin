import torch
import torch.nn as nn
import numpy as np


class WrappedPhaseLoss(nn.Module):
    """ 
    Phase-wrap aware loss, adapted from https://github.com/yxlu-0102/MP-SENet/tree/main
    """
    def __init__(self):
        super(WrappedPhaseLoss, self).__init__()

    def anti_wrapping_function(self, x):
        return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

    def forward(self, target_cplx, predicted_cplx):

        target_p = torch.atan2(torch.imag(target_cplx)+(1e-10), torch.real(target_cplx)+(1e-5))
        predicted_p = torch.atan2(torch.imag(predicted_cplx)+(1e-10), torch.real(predicted_cplx)+(1e-5))
        ip_loss = torch.mean(self.anti_wrapping_function(target_p - predicted_p))
        gd_loss = torch.mean(self.anti_wrapping_function(torch.diff(target_p, dim=1) - torch.diff(predicted_p, dim=1)))
        iaf_loss = torch.mean(self.anti_wrapping_function(torch.diff(target_p, dim=2) - torch.diff(predicted_p, dim=2)))

        return ip_loss + gd_loss + iaf_loss

