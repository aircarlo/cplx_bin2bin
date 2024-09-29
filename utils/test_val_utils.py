import os, sys
import yaml
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from utils.audio_utils_torch import FGLA_custom_torch
from utils.utils import file_parse

class FW_inpainter_VCTK:
    """
    Inpainter class used during the validation cycle.
    It uses "full" stride.
    Full-CUDA implementation
    """
    def __init__(self, params, test_plr):
        self.par = params
        self.test_plr = test_plr
        # STFT class
        self.spectrogram = T.Spectrogram(n_fft=self.par["FFT_SIZE"],
                                         hop_length=self.par["FFT_HOP"],
                                         power=None,
                                         normalized=True).to(self.par["DEVICE"])

        # ISTFT class
        self.inv_spectrogram = T.InverseSpectrogram(n_fft=self.par["FFT_SIZE"],
                                                    hop_length=self.par["FFT_HOP"]) .to(self.par["DEVICE"])

        # resampler class
        self.resampler = T.Resample(self.par["VCTK_SAMPLE_RATE"],
                                    self.par["SAMPLE_RATE"],
                                    resampling_method="sinc_interp_kaiser",
                                    dtype=torch.float32).to(self.par["DEVICE"])



    def infer(self, fname, model, F2T_mode):

        # load the clean waveform and downsample to 16kHz
        y_clean, _ = torchaudio.load(fname)
        y_clean = y_clean.to(self.par['DEVICE'])
        y_clean = self.resampler(y_clean)

        # trim to N*20ms
        len_samples = y_clean.shape[1]
        y_clean = y_clean[:,0:(len_samples//320*320)]

        y_lossy = torch.clone(y_clean)

        # generate pseudo-random gap mask
        # np.random.seed(0) # for reproducibility
        len_frames = len_samples//320 # 320 samples = 1 frame = 20ms @ 16kHz
        frame_mask = list(np.random.choice([0, 1], size=(len_frames,), p=[self.test_plr, 1-self.test_plr]))

        # # load fixed gap mask
        # mask_fname = (fname[:-3] + 'npy').split('/')[-1]
        # frame_mask = np.load(os.path.join(self.par["VAL_MASK_PATH"], mask_fname))

        sample_mask = torch.tensor(np.repeat(frame_mask, 320))
        sample_mask = sample_mask.unsqueeze(0).to(self.par['DEVICE'])
        y_lossy *= sample_mask

        # normalize
        v = torch.var(y_lossy)
        y_lossy /= v

        # compute the lossy complex spectrogram
        S_lossy = self.spectrogram(y_lossy) # .to(self.par["DEVICE"])

        S_inp = torch.zeros_like(S_lossy)

        max_t = S_lossy.shape[2]
        N = max_t // 257
        with torch.no_grad():
            for n in range(N):
                start_bin = n * 257
                end_bin = (n + 1) * 257
                S_inp[0, :, start_bin:end_bin] = model(S_lossy[:, :, start_bin:end_bin].unsqueeze(0))
            S_inp[0, :, max_t - 257:max_t] = model(S_lossy[:, :, max_t - 257:max_t].unsqueeze(0))

        # back to time
        # S_inp = S_inp.detach().cpu()
        if F2T_mode=='ISTFT':
            y_inpainted = self.inv_spectrogram(S_inp, y_clean.shape[1])
        if F2T_mode=='FGLA': # GRIFFIN-LIM with custom initialization
            S_pha = S_inp.angle()
            y_inpainted = FGLA_custom_torch(torch.abs(S_inp),
                                            n_iter=self.par["GLA_ITERS"],
                                            length=y_clean.shape[1],
                                            init="custom", # custom or random
                                            init_tensor=S_pha) 

        # de-normalize
        y_lossy *= v

        # amplitude norm
        y_inpainted = y_inpainted/torch.max(y_inpainted)*torch.max(y_lossy)

        # return torch.squeeze(y_clean), torch.squeeze(y_lossy), torch.squeeze(y_inpainted)
        return y_clean.detach().cpu(), y_lossy.detach().cpu(), y_inpainted.detach().cpu()


class FW_inpainter_VCTK_custom:
    """
    Inpainter class with custom context (i.e. adjustable stride)
    """
    def __init__(self, params, test_plr, stride = None):
        self.par = params
        self.test_plr = test_plr
        self.stride = self.par["FW_STRIDE"] if stride == None else stride
        print(f'STRIDE = {self.stride}')

        # STFT class
        self.spectrogram = T.Spectrogram(n_fft=self.par["FFT_SIZE"],
                                         hop_length=self.par["FFT_HOP"],
                                         power=None,
                                         normalized=True)

        # ISTFT class
        self.inv_spectrogram = T.InverseSpectrogram(n_fft=self.par["FFT_SIZE"],
                                                    hop_length=self.par["FFT_HOP"])

        # resampler class
        self.resampler = T.Resample(self.par["VCTK_SAMPLE_RATE"],
                                    self.par["SAMPLE_RATE"],
                                    resampling_method="sinc_interp_kaiser",
                                    dtype=torch.float32)


    def infer(self, fname, model, F2T_mode):

        # load the clean waveform and downsample to 16kHz
        y_clean, _ = torchaudio.load(fname)
        y_clean = self.resampler(y_clean)
        # stride_ms = int((self.par["FW_STRIDE"]/5)*320)
        stride_ms = int((self.stride/5)*320)
        pre_pad = 16384-stride_ms
        y_clean = F.pad(y_clean, pad=(pre_pad, 0), mode='constant', value=y_clean[0,0]) # append a pre-pad of zeros

        # trim to N*20ms
        len_samples = y_clean.shape[1]
        y_clean = y_clean[:,0:(len_samples//320*320)]

        y_lossy = torch.clone(y_clean)

        # generate random gap mask (deprecated)
        len_frames = len_samples//320 # 320 samples = 1 frame = 20ms @ 16kHz
        frame_mask = list(np.random.choice([0, 1], size=(len_frames,), p=[self.test_plr, 1-self.test_plr]))
        
        sample_mask = torch.tensor(np.repeat(frame_mask, 320))
        sample_mask = sample_mask.unsqueeze(0)
        y_lossy *= sample_mask

        bin_mask = torch.tensor(np.repeat(frame_mask, 5))
        bin_mask = bin_mask.unsqueeze(0)

        # normalize
        v = torch.var(y_lossy)
        y_lossy /= v

        # compute the lossy complex spectrogram
        S_lossy = self.spectrogram(y_lossy).to(self.par["DEVICE"])
        S_inp = torch.clone(S_lossy)

        ctc = range(0, S_lossy.shape[2]-257, self.par["FW_STRIDE"])
        max_t = S_lossy.shape[2]
        with torch.no_grad():
            for n in ctc:
                start_bin = n
                end_bin = n + 257
                if (1-bin_mask[0,end_bin-self.par["FW_STRIDE"]:end_bin]).sum() > 0.0:
                    S_out = model(S_inp[:, :, start_bin:end_bin].unsqueeze(0)) # [B,1,257,257]
                    S_inp[0, :, end_bin-self.par["FW_STRIDE"]:end_bin] = S_out[0, 0, :, -self.par["FW_STRIDE"]:]

            S_out = model(S_inp[:, :, max_t - 257:max_t].unsqueeze(0))
            S_inp[0, :, max_t-self.par["FW_STRIDE"]:max_t] = S_out[0, 0, :, -self.par["FW_STRIDE"]:]

        # back to time
        S_inp = S_inp.detach().cpu()
        if F2T_mode=='ISTFT':
            y_inpainted = self.inv_spectrogram(S_inp, y_clean.shape[1])
        if F2T_mode=='FGLA': # GRIFFIN-LIM with custom initialization
            S_pha = S_inp.angle()
            y_inpainted = FGLA_custom_torch(torch.abs(S_inp),
                                            n_iter=self.par["GLA_ITERS"],
                                            length=y_clean.shape[1],
                                            init="custom", # custom or random
                                            init_tensor=S_pha) 

        # de-normalize
        y_lossy *= v

        # amplitude norm
        y_inpainted = y_inpainted/torch.max(y_inpainted)*torch.max(y_lossy)

        # return torch.squeeze(y_clean), torch.squeeze(y_lossy), torch.squeeze(y_inpainted)
        # return without pre_pad
        return y_clean[:,pre_pad:], y_lossy[:,pre_pad:], y_inpainted[:,pre_pad:]


class FW_inpainter_PLCC2022:
    """
    Inpainter class for IS_PLCChallenge_2022 dataset.
    It uses "full" stride.
    Full-CUDA implementation
    """
    def __init__(self, params, clean_path, lossy_path, restored_path):
        self.par = params
        self.clean_path = clean_path
        self.lossy_path = lossy_path
        self.restored_path = restored_path

        # STFT class
        self.spectrogram = T.Spectrogram(n_fft=self.par["FFT_SIZE"],
                                         hop_length=self.par["FFT_HOP"],
                                         power=None,
                                         normalized=True).to(self.par["DEVICE"])

        # ISTFT class
        self.inv_spectrogram = T.InverseSpectrogram(n_fft=self.par["FFT_SIZE"],
                                                    hop_length=self.par["FFT_HOP"]) .to(self.par["DEVICE"])

        # resampler class
        self.resampler = T.Resample(self.par["VCTK_SAMPLE_RATE"],
                                    self.par["SAMPLE_RATE"],
                                    resampling_method="sinc_interp_kaiser",
                                    dtype=torch.float32).to(self.par["DEVICE"])

        



    def infer(self, fname, model, F2T_mode):

        # load the clean waveform and downsample to 16kHz
        y_clean, sr_clean = torchaudio.load(os.path.join(self.clean_path, fname))
        # trim to N*20ms
        len_samples = y_clean.shape[1]
        y_clean = y_clean[:,0:(len_samples//320*320)]

        y_lossy, sr_lossy = torchaudio.load(os.path.join(self.lossy_path, fname))
        # trim to N*20ms
        len_samples = y_lossy.shape[1]
        y_lossy = y_lossy[:,0:(len_samples//320*320)]
        y_lossy = y_lossy.to(self.par['DEVICE'])

        # normalize
        v = torch.var(y_lossy)
        y_lossy /= v
        # y_lossy = (y_lossy - torch.min(y_lossy))/(torch.max(y_lossy) - torch.min(y_lossy)) # [0,1] range

        # compute the lossy complex spectrogram
        S_lossy = self.spectrogram(y_lossy) # .to(self.par["DEVICE"])

        S_inp = torch.zeros_like(S_lossy)

        max_t = S_lossy.shape[2]
        N = max_t // 257
        with torch.no_grad():
            for n in range(N):
                start_bin = n * 257
                end_bin = (n + 1) * 257
                S_inp[0, :, start_bin:end_bin] = model(S_lossy[:, :, start_bin:end_bin].unsqueeze(0))
            S_inp[0, :, max_t - 257:max_t] = model(S_lossy[:, :, max_t - 257:max_t].unsqueeze(0))

        # back to time
        # S_inp = S_inp.detach().cpu()
        if F2T_mode=='ISTFT':
            y_inpainted = self.inv_spectrogram(S_inp, y_clean.shape[1])
        if F2T_mode=='FGLA': # GRIFFIN-LIM with custom initialization
            S_pha = S_inp.angle().detach().cpu()
            S_inp = S_inp.detach().cpu()
            y_inpainted = FGLA_custom_torch(torch.abs(S_inp),
                                            n_iter=self.par["GLA_ITERS"],
                                            length=y_clean.shape[1],
                                            init="custom", # custom or random
                                            init_tensor=S_pha) 

        # de-normalize
        y_lossy *= v

        # amplitude norm
        # y_inpainted = y_inpainted/torch.max(y_inpainted)*torch.max(y_lossy)
        y_inpainted = y_inpainted/torch.max(y_inpainted)*torch.max(y_lossy.detach().cpu())

        # return torch.squeeze(y_clean), torch.squeeze(y_lossy), torch.squeeze(y_inpainted)
        return y_clean.detach().cpu(), y_lossy.detach().cpu(), y_inpainted.detach().cpu()


if __name__ == "__main__":
    pass

