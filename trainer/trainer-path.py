import librosa
import torch.autograd as autograd
from torch.autograd import Variable
import random
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch
import soundfile as sf
import math
import time
from torch.distributions import Bernoulli, Categorical
import torch
from scipy.io import savemat
import os
import scipy.io as io
import torch.nn.functional as functional
import torchaudio
from trainer.base_trainer1 import BaseTrainer
from util.utils import compute_STOI, compute_PESQ, overlap_cat
from util.pip import NetFeeder, Resynthesizer, wavNormalize
from scipy.io import savemat

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model1,
            model2,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model1, model2, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
    



    def _train_epoch(self, epoch):
        train_loss = []
        loss_total = 0.0
        loss1_total = 0.0
        loss3_total = 0.0
        num_batchs = len(self.train_data_loader)
        num_index = 0
        with tqdm(total = num_batchs) as pbar:
            for i, (mixture, clean, name) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()

                
                mixture = mixture.to(self.device) #[B, T]
                clean = clean.to(self.device) #[B, T]
                mixture_d = torch.stft(mixture, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).to(self.device)) #[B, F, T, 2]
                mixture_mag, mixture_phase = torchaudio.functional.magphase(mixture_d) # [B, F, T], [B, F, T]
                #mixture_ri = feeder(mixture.type(torch.cuda.FloatTensor)) # For complex spectrum extraction
				
                clean_d = torch.stft(clean, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).to(self.device)) #[B, F, T, 2]
                clean_mag, clean_phase = torchaudio.functional.magphase(clean_d) # [B, F, T], [B, F, T]
				#clean_ri = feeder(clean.type(torch.cuda.FloatTensor)
#################################################################################################################################################

##############################   initialize   #########################################################
                clean_mag = clean_mag.type(torch.cuda.FloatTensor)
##################################################################################################################################################	
                probs_t, probs_f = self.model1(mixture_mag.type(torch.cuda.FloatTensor))
                policy_t_map = probs_t.data.clone()
                policy_t_map = Variable(policy_t_map)
                distr_t = Bernoulli(probs_t)
                policy_t = distr_t.sample()
                policy_f_map = probs_f.data.clone()
                policy_f_map = Variable(policy_f_map)
                distr_f = Bernoulli(probs_f)
                policy_f = distr_f.sample()
                enhanced_mag_map = self.model2.full_forward(mixture_mag.type(torch.cuda.FloatTensor), policy_t.data, policy_f.data)
                #enhanced = resynthesizer(enhanced, dual_r.type(torch.cuda.FloatTensor)) # For iSTFT 

                loss = self.loss_function(enhanced_mag_map, clean_mag)			

                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()

                num_index += 1

                pbar.update(1)
                
        end_time = time.time()
        

        dl_len = len(self.train_data_loader)
        print("loss:", loss_total / dl_len)

        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        num_batchs = len(self.validation_data_loader)
        loss_total = 0.0
        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        num_index = 0
        with tqdm(total = num_batchs) as pbar:
            for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
                assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
                name = name[0]
                padded_length = 0
                mixture = mixture.to(self.device) #[B, T]

                clean = clean.to(self.device) #[B, T]

				
                mixture_d = torch.stft(mixture, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).to(self.device)) #[B, F, T, 2]
                mixture_mag, mixture_phase = torchaudio.functional.magphase(mixture_d) # [B, F, T], [B, F, T]
			    #mixture_ri = feeder(mixture.type(torch.cuda.FloatTensor)) # For complex spectrum extraction			
                              
                clean_d = torch.stft(clean, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).to(self.device)) #[B, F, T, 2]
                clean_mag, clean_phase = torchaudio.functional.magphase(clean_d) # [B, F, T], [B, F, T]
                #clean_ri = feeder(clean.type(torch.cuda.FloatTensor)
                clean_mag = clean_mag.type(torch.cuda.FloatTensor)
##################################################################################################################################################
                probs_t, probs_f = self.model1(mixture_mag.type(torch.cuda.FloatTensor))
                policy_t_map = probs_t.data.clone()
                policy_t_map = Variable(policy_t_map)
                distr_t = Bernoulli(probs_t)
                policy_t = distr_t.sample()
                policy_f_map = probs_f.data.clone()
                policy_f_map = Variable(policy_f_map)
                distr_f = Bernoulli(probs_f)
                policy_f = distr_f.sample()
                enhanced_mag = self.model2.full_forward(mixture_mag.type(torch.cuda.FloatTensor), policy_t.data, policy_f.data)
                #enhanced = resynthesizer(enhanced, dual_r.type(torch.cuda.FloatTensor)) # For iSTFT                
				
				loss = self.loss_function(enhanced_mag, clean_mag)
            
                

                loss_total = loss.item()
 

                enhanced_d = torch.cat([
                (enhanced_mag * torch.cos(mixture_phase)).unsqueeze(-1),
                (enhanced_mag * torch.sin(mixture_phase)).unsqueeze(-1)
                ], dim=-1)  # [B, F, T, 2]
                enhanced_d = enhanced_d.type(torch.cuda.DoubleTensor)
                window = torch.hann_window(512).type(torch.cuda.DoubleTensor)
                enhanced = torch.istft(enhanced_d, n_fft=512, hop_length=256, win_length=512, window=window, length=mixture.shape[1])
                mixture = mixture.detach().squeeze(0).cpu().numpy()
                clean = clean.detach().squeeze(0).cpu().numpy()
                enhanced = enhanced.detach().squeeze(0).cpu().numpy()


                assert len(mixture) == len(enhanced) == len(clean)
          
                pbar.update(1)
  

        print("loss:", loss_total/num_batchs)
        score = loss_total
        
        return score