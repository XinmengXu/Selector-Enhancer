import torch
import torch.nn as nn
from typing import Tuple, Dict
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()
def cross_entropy():
    return torch.nn.CrossEntropyLoss()



class NonCausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.conv_block_1 = NonCausalConvBlock(1, 16)
        self.conv_block_2 = NonCausalConvBlock(16, 32)
        self.conv_block_3 = NonCausalConvBlock(32, 64)
        self.conv_block_4 = NonCausalConvBlock(64, 128)
        self.conv_block_5 = NonCausalConvBlock(128, 256)
        self.l1 = nn.l1_loss
		
    def forward(self, n, p, c):
        n = n.unsqueeze(1)
        p = p.unsqueeze(1)
        c = c.unsqueeze(1)
        ###################################
        n1 = self.conv_block_1(n)
        n2 = self.conv_block_2(n1)
        n3 = self.conv_block_3(n2)
        n4 = self.conv_block_4(n3)
        n5 = self.conv_block_5(n4)
	    ##################################
        
	    ###################################
        p1 = self.conv_block_1(p)
        p2 = self.conv_block_2(p1)
        p3 = self.conv_block_3(p2)
        p4 = self.conv_block_4(p3)
        p5 = self.conv_block_5(p4)
	    ##################################

	    ###################################
        c1 = self.conv_block_1(c)
        c2 = self.conv_block_2(c1)
        c3 = self.conv_block_3(c2)
        c4 = self.conv_block_4(c3)
        c5 = self.conv_block_5(c4)
	    ##################################        
        d_np1 = self.l1(p1, n1)
        d_np2 = self.l1(p2, n2)
        d_np3 = self.l1(p3, n3)
        d_np4 = self.l1(p4, n4)
        d_np5 = self.l1(p5, n5)
		
        d_cp1 = self.l1(p1, c1)
        d_cp2 = self.l1(p2, c2)
        d_cp3 = self.l1(p3, c3)
        d_cp4 = self.l1(p4, c4)
        d_cp5 = self.l1(p5, c5)
	    ######################################
        loss1 = 1/32 * d_cp1/(d_np1 + 1e-7) + 1/16 * d_cp2/(d_np2 + 1e-7) + 1/8 * d_cp3/(d_np3 + 1e-7) + 1/4 * d_cp4/(d_np4 + 1e-7) + d_cp5/(d_np5 + 1e-7)
        loss2 = self.l1(p, c)
        loss = loss2 + 0.9 * loss1
        return loss
        