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
from util.utils import compute_STOI, compute_PESQ, overlap_cat, load_checkpoint
plt.switch_backend('agg')
from scipy.io import savemat

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function,optimizer)
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

				
                clean_d = torch.stft(clean, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).to(self.device)) #[B, F, T, 2]
                clean_mag, clean_phase = torchaudio.functional.magphase(clean_d) # [B, F, T], [B, F, T]
#################################################################################################################################################

##############################   initialize   #########################################################
                clean_mag = clean_mag.type(torch.cuda.FloatTensor)
##################################################################################################################################################	            
                enhanced_mag, l1, l2, l3, n1, n2, n3, p1, p2, p3, q1, q2, q3 = self.model.forward(mixture_mag.type(torch.cuda.FloatTensor)) 
             
                #loss1 = self.loss_function(enhanced_mag_map, clean_mag)
                #
                reward_n1 = get_reward(n1)
                reward_n2 = get_reward(n2)
                reward_n3 = get_reward(n3)
                reward_l1 = get_reward(l1)
                reward_l2 = get_reward(l2)
                reward_l3 = get_reward(l3)

                advantage_n1 = reward_n1.mean()            
                advantage_n2 = reward_n2.mean()	
                advantage_n3 = reward_n3.mean()
                advantage_l1 = reward_l1.mean()            
                advantage_l2 = reward_l2.mean()	
                advantage_l3 = reward_l3.mean()	
				
##################################################################################################################################################

##################################################################################################################################################
                loss1 = p1.log_prob(l1).sum() * Variable(advantage_l1)
                loss2 = p2.log_prob(l2).sum() * Variable(advantage_l2)
                loss3 = p3.log_prob(l3).sum() * Variable(advantage_l3)
                loss4 = 0.001 * (loss1 + loss2 + loss3)/3
                loss5 = q1.log_prob(n1).sum() * Variable(advantage_n1)
                loss6 = q2.log_prob(n2).sum() * Variable(advantage_n2)
                loss7 = q3.log_prob(n3).sum() * Variable(advantage_n3)
                loss8 = 0.001 * (loss5 + loss6 + loss7)/3
                loss9 = self.loss_function(enhanced_mag, clean_mag)
               
                # print(Variable(reward_map))
                loss = loss4 + loss8 + loss9	
##################################################################################################################################################
##################################################################################################################################################
                #loss3 = 0.1*-distr.log_prob(policy).sum()* Variable(advantage).sum().abs()
                #print(-distr.log_prob(policy).mean()*Variable(advantage))
                #loss = Variable(advantage.mean()) + loss1 
							
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
			
                              
                clean_d = torch.stft(clean, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).to(self.device)) #[B, F, T, 2]
                clean_mag, clean_phase = torchaudio.functional.magphase(clean_d) # [B, F, T], [B, F, T]

##############################   initialize   #########################################################
##############################   initialize   #########################################################
                clean_mag = clean_mag.type(torch.cuda.FloatTensor)
##################################################################################################################################################            

                enhanced_mag, l1, l2, l3, n1, n2, n3, p1, p2, p3, q1, q2, q3  = self.model.forward(mixture_mag.type(torch.cuda.FloatTensor)) 		
                reward_n1 = get_reward(n1)
                reward_n2 = get_reward(n2)
                reward_n3 = get_reward2(n3)
                reward_l1 = get_reward(l1)
                reward_l2 = get_reward(l2)
                reward_l3 = get_reward2(l3)

                advantage_n1 = reward_n1.mean()            
                advantage_n2 = reward_n2.mean()	
                advantage_n3 = reward_n3.mean()
                advantage_l1 = reward_l1.mean()            
                advantage_l2 = reward_l2.mean()	
                advantage_l3 = reward_l3.mean()	
				
##################################################################################################################################################

##################################################################################################################################################
                loss1 = p1.log_prob(l1).sum() * Variable(advantage_l1)
                loss2 = p2.log_prob(l2).sum() * Variable(advantage_l2)
                loss3 = p3.log_prob(l3).sum() * Variable(advantage_l3)
                loss4 = 0.001 * (loss1 + loss2 + loss3)/3
                loss5 = q1.log_prob(n1).sum() * Variable(advantage_n1)
                loss6 = q2.log_prob(n2).sum() * Variable(advantage_n2)
                loss7 = q3.log_prob(n3).sum() * Variable(advantage_n3)
                loss8 = 0.001 * (loss5 + loss6 + loss7)/3
                loss9 = self.loss_function(enhanced_mag, clean_mag)
               
                # print(Variable(reward_map))
                loss = loss4 + loss8 + loss9		            
                loss_total = loss.item()
                """=== === === start overlap enhancement === === ==="""
                # mixture_mag = mixture_mag[None, :, :, :] # [1, F, T] => [1, 1, F, T], 多一个维度是为了 unfold 			
                #num_index += 1				

                # """=== === === end overlap enhancement === === ==="""

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

                # # Visualize audio
                

                # # Visualize waveform
                

                # # Visualize spectrogram

                output_path = os.path.join('H:/data3/r3/', f"{name}.wav")
                librosa.output.write_wav(output_path, enhanced, sr=16000) 
                # Metric
          
                pbar.update(1)
        #get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)

        print("loss:", loss_total/num_batchs)
        score = loss_total
        
        return score

def get_reward(policy):


    reward = -0.08 * policy
   

    return reward
	
def get_reward2(policy, loss):

    if loss >= 0.06:
        reward = -0.08 * policy - loss
    else:
        reward = -0.08 * policy - (loss/0.06) * loss
    return reward