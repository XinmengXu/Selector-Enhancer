import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from thop import profile
from torchstat import stat
from ptflops import get_model_complexity_info

class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        dim1,
        dim2,
        dropout = 0.5,
    ):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv1d(dim1, dim1, kernel_size=1),
            nn.BatchNorm1d(dim1))		
        self.a = nn.Sequential(
            nn.Conv1d(dim1, dim1, kernel_size=1),
            nn.BatchNorm1d(dim1))	
        self.b = nn.Sequential(
            nn.Conv1d(dim1, dim1, kernel_size=1),
            nn.BatchNorm1d(dim1))					
        self.o = nn.Sequential(
            nn.Conv1d(dim1, dim1, kernel_size=1),
            nn.BatchNorm1d(dim1))			
        self.softmax = nn.Softmax(dim=-1)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, context = None, mask = None, context_mask = None):
 
        b, c, w, j = x.size()
        t = x.reshape(b*w,c,j)
        f = x.reshape(b*j,c,w)
        ##########for time axis#############
        q_t = self.g(t)
        k_t = self.a(t)
        v_t = self.b(t)
        atten_t = self.softmax(torch.matmul(q_t, k_t.permute(0, 2, 1)))
        x_t = torch.matmul(atten_t, v_t)
        ###########for frequency axis###############		
        q_f = self.g(f)
        k_f = self.a(f)
        v_f = self.b(f)
        atten_f = self.softmax(torch.matmul(q_f, k_f.permute(0, 2, 1)))
        x_f = torch.matmul(atten_f, v_f)

        xn = self.o(x_t).reshape(b, c, w, j) + self.o(x_f).reshape(b, c, w, j) + x
        return xn


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


class NonCausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
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

class LocalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))		
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1) 		

    def forward(self, x):

        x1 = self.conv2(self.conv1(x))
        x2 = self.conv2(self.conv1(self.conv2(self.conv1(x))))
        x3 = torch.cat([x1, x2], dim = 1)
        x3_1 = self.conv2(x3)
        x3_2 = self.conv2(x3)
        x3 = self.sigmoid(x3_1)*x1 + self.sigmoid(x3_2)*x2
        # x3 = self.conv2(self.conv1(x)) 
        # x3 = self.sigmoid(x3) * x		
        return x3
    
class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 4),
            stride=(4, 62),
            padding=(0, 0)
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

	       

class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = NonCausalConvBlock(1, 16)
        self.conv_block_2 = NonCausalConvBlock(16, 32)
        self.conv_block_3 = NonCausalConvBlock(32, 64)
        self.conv_block_4 = NonCausalConvBlock(64, 128)
        self.conv_block_5 = NonCausalConvBlock(128, 256)
        self.sigmoid = nn.Sigmoid()
        self.LocalAttention = LocalAttention()
        self.NonLocalAttention = MultiHeadCrossAttention(256, 256)
        self.tran_conv_block_1 = NonCausalTransConvBlock(256, 128)
        self.tran_conv_block_2 = NonCausalTransConvBlock(256, 64)
        self.tran_conv_block_3 = NonCausalTransConvBlock(128, 32)
        self.tran_conv_block_4 = NonCausalTransConvBlock(64, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = NonCausalTransConvBlock(32, 1, is_last=True)	
        self.logit1 = nn.Linear(25, 251)
        self.up = TransConvBlock(256, 128)
	
    def full_forward(self, x_o, prob_t, prob_f):
        prob_t =  prob_t.chunk(3, dim = 1)
        prob_f =  prob_f.chunk(3, dim = 1)
        x_o = x_o.unsqueeze(1)

        s = self.conv_block_1(x_o)
        s1 = self.conv_block_2(s)
        s2 = self.conv_block_3(s1) 
        s3 = self.conv_block_4(s2)

#########################################################################################		
        # Dynamic block 1
        prob_t_1 = prob_t[0].chunk(2, dim = -1)
        prob_f_1 = prob_f[0].chunk(2, dim = -2)
        prob_n = torch.matmul(prob_f_1[0], prob_t_1[0])
        prob_l = torch.matmul(prob_f_1[1], prob_t_1[1]) 
        n_mask = self.up(prob_n)
        l_mask = self.up(prob_l)

        s_n = self.conv_block_5(s3 * n_mask)
        x_n = self.NonLocalAttention(s_n, s_n)
        s_l = self.conv_block_5(s3 * l_mask)
        x_l = self.LocalAttention(s_l)	
        x = self.tran_conv_block_1(x_n + x_l) + s3
       		
###########################################################################################
        # Dynamic block 2
        prob_t_1 = prob_t[1].chunk(2, dim = -1)
        prob_f_1 = prob_f[1].chunk(2, dim = -2)
        prob_n = torch.matmul(prob_f_1[0], prob_t_1[0])
        prob_l = torch.matmul(prob_f_1[1], prob_t_1[1]) 
        n_mask = self.up(prob_n)
        l_mask = self.up(prob_l)

        s_n = self.conv_block_5(x * n_mask)
        x_n = self.NonLocalAttention(s_n, s_n)
        s_l = self.conv_block_5(x * l_mask)
        x_l = self.LocalAttention(s_l)	
        x1 = self.tran_conv_block_1(x_n + x_l) + x
###############################################################################################
        # Dynamic block 2
        prob_t_1 = prob_t[2].chunk(2, dim = -1)
        prob_f_1 = prob_f[2].chunk(2, dim = -2)
        prob_n = torch.matmul(prob_f_1[0], prob_t_1[0])
        prob_l = torch.matmul(prob_f_1[1], prob_t_1[1]) 
        n_mask = self.up(prob_n)
        l_mask = self.up(prob_l)

        s_n = self.conv_block_5(x1 * n_mask)
        x_n = self.NonLocalAttention(s_n, s_n)
        s_l = self.conv_block_5(x1 * l_mask)
        x_l = self.LocalAttention(s_l)	
        x = self.tran_conv_block_1(x_n + x_l) + x1
###########################################################################################
        r = self.tran_conv_block_2(torch.cat([x, s3], 1))
        r = self.tran_conv_block_3(torch.cat([r, s2], 1))
        r = self.tran_conv_block_4(torch.cat([r, s1], 1))
        r = self.tran_conv_block_5(torch.cat([r, s], 1))
        r = r.squeeze()	
		
        return r
###################################################################################################		


    def forward(self, x_o):
        #x_o = x_o.unsqueeze(1)

        x0 = self.conv_block_1(x_o)
        x0 = self.conv_block_2(x0)
        x0 = self.conv_block_3(x0)

################################################
        # path for local attention
        xo_1 = self.LocalAttention(x0)

        gate = self.sigmoid(xo_1)
        if gate.mean() >= 0.5:
           x_local = xo_1
           print(1)
        else:
           x_local = x0
##############################################				
        #path for non-local attention				
        xo_2 = self.NonLocalAttention(x0, x0)
        gate = self.sigmoid(xo_2)
        if gate.mean() >= 0.5:
           x_nonlocal = xo_2
           print(1)
        else:
           x_nonlocal = x0
###############################################
        x = x_local + x_nonlocal
        x = self.tran_conv_block_3(x)
        x = self.tran_conv_block_4(x)
        xa = self.tran_conv_block_5(x)	
	
###################################################################################################	
###################################################################################################	
        x0 = self.conv_block_1(xa)
        x0 = self.conv_block_2(x0)
        x0 = self.conv_block_3(x0)

################################################
        xo_1 = self.LocalAttention(x0)

        gate = self.sigmoid(xo_1)
        if gate.mean() >= 0.5:
           x_local = xo_1
           print(1)
        else:
           x_local = x0
##############################################				
        #path for non-local attention				
        xo_2 = self.NonLocalAttention(x0, x0)
        gate = self.sigmoid(xo_2)
        if gate.mean() >= 0.5:
           x_nonlocal = xo_2
           print(1)
        else:
           x_nonlocal = x0
###############################################
        x = x_local + x_nonlocal
        x = self.tran_conv_block_3(x)
        x = self.tran_conv_block_4(x)
        xb = self.tran_conv_block_5(x)
##########################################################################################	
##########################################################################################	  
        x0 = self.conv_block_1(xb)
        x0 = self.conv_block_2(x0)
        x0 = self.conv_block_3(x0)

################################################
        # path for local attention
        xo_1 = self.LocalAttention(x0)

        gate = self.sigmoid(xo_1)
        if gate.mean() >= 0.5:
           x_local = xo_1
           print(1)
        else:
           x_local = x0
##############################################				
        #path for non-local attention				
        xo_2 = self.NonLocalAttention(x0, x0)
        gate = self.sigmoid(xo_2)
        if gate.mean() >= 0.5:
           x_nonlocal = xo_2
           print(1)
        else:
           x_nonlocal = x0
###############################################
        x = x_local + x_nonlocal
        x = self.tran_conv_block_3(x)
        x = self.tran_conv_block_4(x)
        x = self.tran_conv_block_5(x)
      
###########################################################################################
        #x = x.squeeze()	
        #xa = xa.squeeze()
        #xb = xb.squeeze()		
        return x
		
		
		
		
		
		
		
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(0, 0)
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

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 4),
            stride=(2, 1),
            padding=(0, 0)
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


class Policy(nn.Module):

    def __init__(self, num_blocks=6):
        super(Policy, self).__init__()

        self.conv_block_1 = ConvBlock(1, 16)
        self.conv_block_2 = ConvBlock(16, 32)
        self.conv_block_3 = ConvBlock(32, 64)
        self.conv_block_4 = ConvBlock(64, 128)
        self.conv_block_5 = ConvBlock2(128, 256)
        self.conv_block_6 = NonCausalConvBlock(16, 32)
        
        self.lstm_layer = nn.LSTM(input_size=5, hidden_size=5, num_layers=2, batch_first=True)
        self.logit = nn.Linear(5, 5)
        self.logit1 = nn.Linear(5, 5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)

        x = self.conv_block_5(x)

        batch_size, n_channels, n_f_bins, n_frame_size = x.shape
		############################################################
        lstm_in = x.reshape(batch_size, n_channels * n_f_bins, n_frame_size)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out_l_1 = self.logit(lstm_out)
        lstm_out_n_1 = self.logit(lstm_out)

        lstm_out_l_1 = lstm_out_l_1.reshape(batch_size, n_channels, n_f_bins, n_frame_size)
        lstm_out_n_1 = lstm_out_n_1.reshape(batch_size, n_channels, n_f_bins, n_frame_size)
	   
        probs_l_1 = F.sigmoid(lstm_out_l_1)
        probs_n_1 = F.sigmoid(lstm_out_n_1)     
        probs_l_t_1 = probs_l_1.mean(-2).unsqueeze(-2)	
        probs_l_f_1 = probs_l_1.mean(-1).unsqueeze(-1)		
		
        probs_n_t_1 = probs_n_1.mean(-2).unsqueeze(-2)
        probs_n_f_1 = probs_n_1.mean(-1).unsqueeze(-1)	

        zero_t = torch.zeros_like(probs_l_t_1)
        zero_f = torch.zeros_like(probs_l_f_1)

        one_t = torch.ones_like(probs_l_t_1)
        one_f = torch.ones_like(probs_l_f_1)
		
        probs_l_t_1 = torch.where(probs_l_t_1>=0.5, one_t, zero_t)
        probs_n_t_1 = torch.where(probs_n_t_1>=0.5, one_t, zero_t)

        probs_l_f_1 = torch.where(probs_l_f_1>=0.5, one_f, zero_f)
        probs_n_f_1 = torch.where(probs_n_f_1>=0.5, one_f, zero_f)			
        probs_t_1 = torch.cat([probs_n_t_1, probs_l_t_1], -1)       		
        probs_f_1 = torch.cat([probs_n_f_1, probs_l_f_1], -2)	
####################################################################


###############################################################
        lstm_out_l_2 = self.logit(lstm_out)
        lstm_out_n_2 = self.logit(lstm_out)

        lstm_out_l_2 = lstm_out_l_2.reshape(batch_size, n_channels, n_f_bins, n_frame_size)
        lstm_out_n_2 = lstm_out_n_2.reshape(batch_size, n_channels, n_f_bins, n_frame_size)
	   
        probs_l_2 = F.sigmoid(lstm_out_l_2)
        probs_n_2 = F.sigmoid(lstm_out_n_2)     
        probs_l_t_2 = probs_l_2.mean(-2).unsqueeze(-2)	
        probs_l_f_2 = probs_l_2.mean(-1).unsqueeze(-1)		
		
        probs_n_t_2 = probs_n_2.mean(-2).unsqueeze(-2)
        probs_n_f_2 = probs_n_2.mean(-1).unsqueeze(-1)	
		
        probs_l_t_2 = torch.where(probs_l_t_2>=0.5, one_t, zero_t)
        probs_n_t_2 = torch.where(probs_n_t_2>=0.5, one_t, zero_t)

        probs_l_f_2 = torch.where(probs_l_f_2>=0.5, one_f, zero_f)
        probs_n_f_2 = torch.where(probs_n_f_2>=0.5, one_f, zero_f)			
        probs_t_2 = torch.cat([probs_n_t_2, probs_l_t_2], -1)       		
        probs_f_2 = torch.cat([probs_n_f_2, probs_l_f_2], -2)				
#########################################################################

        lstm_out_l_3 = self.logit(lstm_out)
        lstm_out_n_3 = self.logit(lstm_out)

        lstm_out_l_3 = lstm_out_l_3.reshape(batch_size, n_channels, n_f_bins, n_frame_size)
        lstm_out_n_3 = lstm_out_n_3.reshape(batch_size, n_channels, n_f_bins, n_frame_size)
	   
        probs_l_3 = F.sigmoid(lstm_out_l_3)
        probs_n_3 = F.sigmoid(lstm_out_n_3)     
        probs_l_t_3 = probs_l_3.mean(-2).unsqueeze(-2)	
        probs_l_f_3 = probs_l_3.mean(-1).unsqueeze(-1)		
		
        probs_n_t_3 = probs_n_3.mean(-2).unsqueeze(-2)
        probs_n_f_3 = probs_n_3.mean(-1).unsqueeze(-1)	

		
        probs_l_t_3 = torch.where(probs_l_t_3>=0.5, one_t, zero_t)
        probs_n_t_3 = torch.where(probs_n_t_3>=0.5, one_t, zero_t)

        probs_l_f_3 = torch.where(probs_l_f_3>=0.5, one_f, zero_f)
        probs_n_f_3 = torch.where(probs_n_f_3>=0.5, one_f, zero_f)			
        probs_t_3 = torch.cat([probs_n_t_3, probs_l_t_3], -1)       		
        probs_f_3 = torch.cat([probs_n_f_3, probs_l_f_3], -2)		
########################################################################
        probs_f = torch.cat([probs_f_1, probs_f_2, probs_f_3], 1)
        probs_t = torch.cat([probs_t_1, probs_t_2, probs_t_3], 1)
        return probs_t, probs_f
		


if __name__ == '__main__':
    layer = Policy()
    a = torch.rand(1, 257, 251)
    b, c = layer(a)
    print(b.size())