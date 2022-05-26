import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from thop import profile
from torchstat import stat
from ptflops import get_model_complexity_info
from torch.distributions import Bernoulli, Categorical
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
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(1, 2),
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
            kernel_size=(1, 1),
            stride=(1, 1),
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
        self.up = nn.Upsample(size=(7, 251), mode='nearest')
		
        self.lstm_layer = nn.LSTM(input_size=252, hidden_size=252, num_layers=2, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(252,63),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.Linear(63,7)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(252,150),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.Linear(150,25)
        )
        self.conv_asn1 = ConvBlock(128, 64)
        self.conv_asn2 = ConvBlock(64, 32)
        self.conv_asn3 = ConvBlock(32, 16)
        self.conv_asn4 = ConvBlock2(16, 1)		
    def full_forward(self, x_o):
        x_o = x_o.unsqueeze(1)

        s = self.conv_block_1(x_o)
        s1 = self.conv_block_2(s)
        s2 = self.conv_block_3(s1) 
        s3 = self.conv_block_4(s2)

#########################################################################################		
        # Dynamic block 1
        #output of first dynamic block
        ##################### selector ###########################################
        a = self.conv_asn1(s3)
        a = self.conv_asn2(a)
        a = self.conv_asn3(a) # size: b, 16, 9, 31
        a = self.conv_asn4(a) # size: b, 1, 9, 31 
        batch_size, n_channels, n_f_bins, n_frame_size = a.shape
        lstm_in = a.reshape(batch_size, n_channels, n_f_bins * n_frame_size)
        # for non-local attention
        lstm_out1, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out1).unsqueeze(-1)
        a_t = self.fc2(lstm_out1).unsqueeze(-2)
        n_mask_1 = self.sigmoid(torch.matmul(a_f, a_t))
        zero = torch.zeros_like(n_mask_1)
        one = torch.ones_like(n_mask_1)
        n_mask_1 = torch.where(n_mask_1<0.5000, n_mask_1, one)
        n_mask_1 = torch.where(n_mask_1>=0.5000, n_mask_1, zero)
        # for local attention
        lstm_out2, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out2).unsqueeze(-1)
        a_t = self.fc2(lstm_out2).unsqueeze(-2)
        l_mask_1 = self.sigmoid(torch.matmul(a_f, a_t))
        l_mask_1 = torch.where(l_mask_1<0.5000, l_mask_1, one)
        l_mask_1 = torch.where(l_mask_1>=0.5000, l_mask_1, zero)
        ###############################################################
        n_mask_1 = torch.randint(0, 2, [1, 1, 7, 25], dtype=torch.float).to(s.device)
        l_mask_1 = torch.randint(0, 2, [1, 1, 7, 25], dtype=torch.float).to(s.device)
        n_mask_u1 = self.up(n_mask_1)
        l_mask_u1 = self.up(l_mask_1)
        mask_1 = torch.cat([n_mask_1, l_mask_1],1)

        db1 = self.conv_block_5(s3)
        db_l_1 = db1 * l_mask_u1
        db_n_1 = db1 * n_mask_u1
        x_n = self.NonLocalAttention(db_n_1, db_n_1)
        x_l = self.LocalAttention(db_l_1)
        		
        x0 = self.tran_conv_block_1(x_n + x_l) + s3
       		
###########################################################################################
        # Dynamic block 2
        ##################### selector ###########################################
        a = self.conv_asn1(x0)
        a = self.conv_asn2(a)
        a = self.conv_asn3(a) # size: b, 16, 9, 31
        a = self.conv_asn4(a) # size: b, 1, 9, 31 
        batch_size, n_channels, n_f_bins, n_frame_size = a.shape
        lstm_in = a.reshape(batch_size, n_channels, n_f_bins * n_frame_size)
        # for non-local attention
        lstm_out1, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out1).unsqueeze(-1)
        a_t = self.fc2(lstm_out1).unsqueeze(-2)
        n_mask_2 = self.sigmoid(torch.matmul(a_f, a_t))

        n_mask_2 = torch.where(n_mask_2<0.5000, n_mask_2, one)
        n_mask_2 = torch.where(n_mask_2>=0.5000, n_mask_2, zero)
        # for local attention
        lstm_out2, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out2).unsqueeze(-1)
        a_t = self.fc2(lstm_out2).unsqueeze(-2)
        l_mask_2 = self.sigmoid(torch.matmul(a_f, a_t))
        l_mask_2 = torch.where(l_mask_2<0.5000, l_mask_2, one)
        l_mask_2 = torch.where(l_mask_2>=0.5000, l_mask_2, zero)
		#########################################################################
        n_mask_2 = torch.randint(0, 2, [1, 1, 7, 25], dtype=torch.float).to(s.device)
        l_mask_2 = torch.randint(0, 2, [1, 1, 7, 25], dtype=torch.float).to(s.device)
        n_mask_u2 = self.up(n_mask_2)
        l_mask_u2 = self.up(l_mask_2)
        mask_2 = torch.cat([n_mask_2, l_mask_2],1)

        db2 = self.conv_block_5(x0)
        db_l_2 = db2 * l_mask_u2
        db_n_2 = db2 * n_mask_u2
        x_n = self.NonLocalAttention(db_n_2, db_n_2)
        x_l = self.LocalAttention(db_l_2)
        x1 = self.tran_conv_block_1(x_n + x_l) + x0
###############################################################################################
        # Dynamic block 3
        ##################### selector ###########################################
        a = self.conv_asn1(x0)
        a = self.conv_asn2(a)
        a = self.conv_asn3(a) # size: b, 16, 9, 31
        a = self.conv_asn4(a) # size: b, 1, 9, 31 
        batch_size, n_channels, n_f_bins, n_frame_size = a.shape
        lstm_in = a.reshape(batch_size, n_channels, n_f_bins * n_frame_size)
        # for non-local attention
        lstm_out1, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out1).unsqueeze(-1)
        a_t = self.fc2(lstm_out1).unsqueeze(-2)
        n_mask_3 = self.sigmoid(torch.matmul(a_f, a_t))

        n_mask_3 = torch.where(n_mask_3<0.5000, n_mask_3, one)
        n_mask_3 = torch.where(n_mask_3>=0.5000, n_mask_3, zero)
        # for local attention
        lstm_out2, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out2).unsqueeze(-1)
        a_t = self.fc2(lstm_out2).unsqueeze(-2)
        l_mask_3 = self.sigmoid(torch.matmul(a_f, a_t))
        l_mask_3 = torch.where(l_mask_3<0.5000, l_mask_3, one)
        l_mask_3 = torch.where(l_mask_3>=0.5000, l_mask_3, zero)

		#########################################################################
        n_mask_3 = torch.randint(0, 2, [1, 1, 7, 25], dtype=torch.float).to(s.device)
        l_mask_3 = torch.randint(0, 2, [1, 1, 7, 25], dtype=torch.float).to(s.device)
        n_mask_u3 = self.up(n_mask_3)
        l_mask_u3 = self.up(l_mask_3)
        mask_3 = torch.cat([n_mask_3, l_mask_3],1)

        db3 = self.conv_block_5(x1)
        db_l_3 = db3 * l_mask_u3
        db_n_3 = db3 * n_mask_u3
        x_n = self.NonLocalAttention(db_n_3, db_n_3)
        x_l = self.LocalAttention(db_l_3)	
        x = self.tran_conv_block_1(x_n + x_l) + x1
###########################################################################################
        r = self.tran_conv_block_2(torch.cat([x, s3], 1))
        r = self.tran_conv_block_3(torch.cat([r, s2], 1))
        r = self.tran_conv_block_4(torch.cat([r, s1], 1))
        r = self.tran_conv_block_5(torch.cat([r, s], 1))
        r = r.squeeze(1)	
		
        return r, mask_1, mask_2, mask_3
###################################################################################################		


    def forward(self, x_o):
        x_o = x_o.unsqueeze(1)

        s = self.conv_block_1(x_o)
        s1 = self.conv_block_2(s)
        s2 = self.conv_block_3(s1) 
        s3 = self.conv_block_4(s2)

#########################################################################################		
        # Dynamic block 1
        ##################### selector ###########################################
        a = self.conv_asn1(s3)
        a = self.conv_asn2(a)
        a = self.conv_asn3(a) # size: b, 16, 9, 31
        a = self.conv_asn4(a) # size: b, 1, 9, 31 
        batch_size, n_channels, n_f_bins, n_frame_size = a.shape
        lstm_in = a.reshape(batch_size, n_channels, n_f_bins * n_frame_size)
        # for non-local attention
        lstm_out1, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out1).unsqueeze(-1)
        a_t = self.fc2(lstm_out1).unsqueeze(-2)
        n_mask_1 = self.sigmoid(torch.matmul(a_f, a_t))
        n_mask_1 = 0.8 * n_mask_1 + (1 - n_mask_1) * 0.2   
        distr_n1 = Bernoulli(n_mask_1)		
        n_mask_1 = distr_n1.sample()		
        # for local attention
        lstm_out2, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out2).unsqueeze(-1)
        a_t = self.fc2(lstm_out2).unsqueeze(-2)
        l_mask_1 = self.sigmoid(torch.matmul(a_f, a_t))
        l_mask_1 = 0.8 * l_mask_1 + (1 - l_mask_1) * 0.2   
        distr_l1 = Bernoulli(l_mask_1)		
        l_mask_1 = distr_l1.sample()		
		#########################################################################

        n_mask_u1 = self.up(n_mask_1)
        l_mask_u1 = self.up(l_mask_1)
        

        db1 = self.conv_block_5(s3)
        db_l_1 = db1 * l_mask_u1
        db_n_1 = db1 * n_mask_u1
        x_n = self.NonLocalAttention(db_n_1, db_n_1)
        x_l = self.LocalAttention(db_l_1)
        		
        x0 = self.tran_conv_block_1(x_n + x_l) + s3
       		
###########################################################################################
        # Dynamic block 2
        ##################### selector ###########################################
        a = self.conv_asn1(x0)
        a = self.conv_asn2(a)
        a = self.conv_asn3(a) # size: b, 16, 9, 31
        a = self.conv_asn4(a) # size: b, 1, 9, 31 
        batch_size, n_channels, n_f_bins, n_frame_size = a.shape
        lstm_in = a.reshape(batch_size, n_channels, n_f_bins * n_frame_size)
        # for non-local attention
        lstm_out1, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out1).unsqueeze(-1)
        a_t = self.fc2(lstm_out1).unsqueeze(-2)
        n_mask_2 = self.sigmoid(torch.matmul(a_f, a_t))
        n_mask_2 = 0.8 * n_mask_2 + (1 - n_mask_2) * 0.2   
        distr_n2 = Bernoulli(n_mask_2)		
        n_mask_2 = distr_n2.sample()		
        # for local attention
        lstm_out2, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out2).unsqueeze(-1)
        a_t = self.fc2(lstm_out2).unsqueeze(-2)
        l_mask_2 = self.sigmoid(torch.matmul(a_f, a_t))
        l_mask_2 = 0.8 * l_mask_2 + (1 - l_mask_2) * 0.2   
        distr_l2 = Bernoulli(l_mask_2)		
        l_mask_2 = distr_l2.sample()		
		#########################################################################
        n_mask_u2 = self.up(n_mask_2)
        l_mask_u2 = self.up(l_mask_2)

        db2 = self.conv_block_5(x0)
        db_l_2 = db2 * l_mask_u2
        db_n_2 = db2 * n_mask_u2
        x_n = self.NonLocalAttention(db_n_2, db_n_2)
        x_l = self.LocalAttention(db_l_2)
        x1 = self.tran_conv_block_1(x_n + x_l) + x0
###############################################################################################
        # Dynamic block 3
        # Dynamic block 2
        ##################### selector ###########################################
        a = self.conv_asn1(x0)
        a = self.conv_asn2(a)
        a = self.conv_asn3(a) # size: b, 16, 9, 31
        a = self.conv_asn4(a) # size: b, 1, 9, 31 
        batch_size, n_channels, n_f_bins, n_frame_size = a.shape
        lstm_in = a.reshape(batch_size, n_channels, n_f_bins * n_frame_size)
        # for non-local attention
        lstm_out1, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out1).unsqueeze(-1)
        a_t = self.fc2(lstm_out1).unsqueeze(-2)
        n_mask_3 = self.sigmoid(torch.matmul(a_f, a_t))
        n_mask_3 = 0.8 * n_mask_3 + (1 - n_mask_3) * 0.2   
        distr_n3 = Bernoulli(n_mask_3)		
        n_mask_3 = distr_n3.sample()		
        # for local attention
        lstm_out2, _ = self.lstm_layer(lstm_in)
        a_f = self.fc1(lstm_out2).unsqueeze(-1)
        a_t = self.fc2(lstm_out2).unsqueeze(-2)
        l_mask_3 = self.sigmoid(torch.matmul(a_f, a_t))
        l_mask_3 = 0.8 * l_mask_3 + (1 - l_mask_3) * 0.2   
        distr_l3 = Bernoulli(l_mask_3)		
        l_mask_3 = distr_l3.sample()		

		#########################################################################
        n_mask_u3 = self.up(n_mask_3)
        l_mask_u3 = self.up(l_mask_3)

        db3 = self.conv_block_5(x1)
        db_l_3 = db3 * l_mask_u3
        db_n_3 = db3 * n_mask_u3
        x_n = self.NonLocalAttention(db_n_3, db_n_3)
        x_l = self.LocalAttention(db_l_3)	
        x = self.tran_conv_block_1(x_n + x_l) + x1
###########################################################################################
        r = self.tran_conv_block_2(torch.cat([x, s3], 1))
        r = self.tran_conv_block_3(torch.cat([r, s2], 1))
        r = self.tran_conv_block_4(torch.cat([r, s1], 1))
        r = self.tran_conv_block_5(torch.cat([r, s], 1))
        r = r.squeeze(1)	
		
        return r, l_mask_1, l_mask_2, l_mask_3, n_mask_1, n_mask_2, n_mask_3, distr_l1, distr_l2, distr_l3, distr_n1, distr_n2, distr_n3, db1, l_mask_u1, n_mask_u1

if __name__ == '__main__':
    layer = CRN()
    a = torch.rand(1, 257, 251)
    b, c, d, d1 = layer.forward(a)
    #print(d)