import torch
import torch.nn.functional as F

import numpy as np
from util.stft import STFT


class NetFeeder(object):
    def __init__(self, device, win_size=512, hop_size=256):
        self.eps = torch.finfo(torch.float32).eps
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, mix):
        real_mix, imag_mix = self.stft.stft(mix)
        feat = torch.stack([real_mix, imag_mix], dim=1)


        return feat


class Resynthesizer(object):
    def __init__(self, device, win_size=512, hop_size=256):
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, est, mix):
        sph_est = self.stft.istft(est)
        sph_est = F.pad(sph_est, [0, mix.shape[1]-sph_est.shape[1]])


        return sph_est
		
def wavNormalize(*sigs):
    # sigs is a list of signals to be normalized
    scale = max([np.max(np.abs(sig)) for sig in sigs]) + np.finfo(np.float32).eps
    sigs_norm = [sig / scale for sig in sigs]
    return sigs_norm