import os, sys
import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torch_audiomentations import Compose, Gain, PolarityInversion, PitchShift
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import random


class VCTK_dataset_raw(Dataset):
    """
    Dataset class for raw spectrograms (complex valued).
    It uses a head+tail guard criterion to avoid the extraction of contexts that may contain silence.
    Full-torch-CUDA implementation
    """

    def __init__(self, wav_list, params, mode):
        self.wav_list = wav_list
        self.par = params
        self.mode = mode  # discriminate between 'train' and 'val', according to meta-info
        self.hg, self.tg = 30, 30  # Head guard bins, Tail guard bins

        # STFT class
        self.spectrogram = T.Spectrogram(n_fft=self.par["FFT_SIZE"],
                                         hop_length=self.par["FFT_HOP"],
                                         power=None,
                                         normalized=True).to(self.par["DEVICE"])

        # resampler class
        self.resampler = T.Resample(self.par["VCTK_SAMPLE_RATE"],
                                    self.par["SAMPLE_RATE"],
                                    resampling_method="sinc_interp_kaiser",
                                    dtype=torch.float32).to(self.par["DEVICE"])

        self.augmentation = Compose(transforms=[Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=0.5),
                                                PolarityInversion(p=0.5),
                                                PitchShift(sample_rate=16000, p=0.5)])
                                                
        self.augmentation = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                                     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                                     PitchShift(min_semitones=-4, max_semitones=4, p=0.5)])

    def __len__(self):
        return len(self.wav_list)


    def __getitem__(self, idx):
        # load the clean waveform
        clean_waveform, _ = torchaudio.load(self.wav_list[idx])
        clean_waveform = clean_waveform.to(self.par["DEVICE"])
        clean_waveform = self.resampler(clean_waveform)  # downsample from 48kHz to 16kHz
        clean_waveform = clean_waveform.unsqueeze(0)
        if self.par["USE_AUGMENTATIONS"]:
            clean_waveform = self.augmentation(clean_waveform, sample_rate=self.par["SAMPLE_RATE"])
        # trim to 20ms
        len_samples = clean_waveform.shape[2]        
        clean_waveform = clean_waveform[0,:, 0:(len_samples // 320 * 320)]
        lossy_waveform = torch.clone(clean_waveform)

        # load/generate the gap mask
        if self.mode == 'val':  # for validation I load a fixed gap mask
            mask_fname = (self.wav_list[idx][:-3] + 'npy').split('/')[-1]
            frame_mask = np.load(os.path.join(self.par["VAL_MASK_PATH"], mask_fname))
        else:
            len_frames = len_samples // 320  # 320 samples = 1 frame = 20ms @ 16kHz
            if self.par["MASK_TYPE"] == 'variable':
                # choose a random loss-rate
                current_lossy_rate = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
                frame_mask = np.random.choice([0, 1], size=(len_frames,), p=[current_lossy_rate, 1-current_lossy_rate])
            elif self.par["MASK_TYPE"] == 'fixed':
                # apply a fixed loss-rate
                frame_mask = np.random.choice([0, 1], size=(len_frames,), p=[self.par["LOSSY_RATE"], 1-self.par["LOSSY_RATE"]])
        
        sample_mask = torch.tensor(np.repeat(frame_mask, 320)) # sample_mask: each element of frame_mask is repeated 320 times (dim: len_frames * 320)
        sample_mask = sample_mask.to(self.par["DEVICE"])

        lossy_waveform *= sample_mask.unsqueeze(0)  # (dim: len_frames * 320)

        # normalizations
        clean_waveform /= torch.var(clean_waveform)
        lossy_waveform /= torch.var(lossy_waveform)

        # compute the spectrograms
        S_clean = self.spectrogram(clean_waveform)
        S_lossy = self.spectrogram(lossy_waveform)

        # extract the context
        if S_clean.shape[2] > (257 + self.hg + self.tg):
            start_bin = random.randint(self.hg, S_clean.shape[2] - (257+self.tg))  # added head and tail guards of 30 bins (approx. 120ms) to naively trim silence...
        else:
            start_bin = random.randint(0, S_clean.shape[2] - 257)
        end_bin = start_bin + 257
        S_clean_frame = S_clean[:, :, start_bin:end_bin]
        S_lossy_frame = S_lossy[:, :, start_bin:end_bin]

        return S_lossy_frame, S_clean_frame  # complex output [1, 257, 257]


class PLCDataset_raw(Dataset):
    """
    Dataset class for raw spectrograms (complex valued).
    It uses a head+tail guard criterion to avoid the extraction of contexts that may contain silence.
    Full-CUDA implementation
    """

    def __init__(self, wav_list, params):
        self.wav_list = wav_list
        self.par = params
        self.hg, self.tg = 30, 30  # Head guard bins, Tail guard bins

        # STFT class
        self.spectrogram = T.Spectrogram(n_fft=self.par["FFT_SIZE"],
                                         hop_length=self.par["FFT_HOP"],
                                         power=None,
                                         normalized=True).to(self.par["DEVICE"])

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        # load the clean waveform
        clean_waveform, _ = torchaudio.load(self.wav_list[idx])
        clean_waveform = clean_waveform.to(self.par["DEVICE"])

        # trim to 20ms
        len_samples = clean_waveform.shape[1]        
        clean_waveform = clean_waveform[:, 0:(len_samples // 320 * 320)]

        lossy_waveform = torch.clone(clean_waveform)

        # load/generate the gap mask

        len_frames = len_samples // 320  # 320 samples = 1 frame = 20ms @ 16kHz
        # choose a random loss-rate
        current_lossy_rate = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        frame_mask = np.random.choice([0, 1], size=(len_frames,), p=[current_lossy_rate, 1-current_lossy_rate])
        # apply a fixed loss-rate
        # frame_mask = np.random.choice([0, 1], size=(len_frames,), p=[self.par["LOSSY_RATE"], 1-self.par["LOSSY_RATE"]])
        
        sample_mask = torch.tensor(np.repeat(frame_mask, 320)) # sample_mask: each element of frame_mask is repeated 320 times (dim: len_frames * 320)
        sample_mask = sample_mask.to(self.par["DEVICE"])

        lossy_waveform *= sample_mask.unsqueeze(0)  # (dim: len_frames * 320)

        # normalizations
        clean_waveform /= torch.var(clean_waveform)
        lossy_waveform /= torch.var(lossy_waveform)
        # clean_waveform = (clean_waveform - torch.min(clean_waveform))/(torch.max(clean_waveform) - torch.min(clean_waveform)) # [0,1] range
        # lossy_waveform = (lossy_waveform - torch.min(lossy_waveform))/(torch.max(lossy_waveform) - torch.min(lossy_waveform)) # [0,1] range

        # compute the spectrograms
        S_clean = self.spectrogram(clean_waveform)
        S_lossy = self.spectrogram(lossy_waveform)

        # extract the context
        if S_clean.shape[2] > (257 + self.hg + self.tg):
            start_bin = random.randint(self.hg, S_clean.shape[2] - (257+self.tg))  # added head and tail guards of 30 bins (approx. 120ms) to naively trim silence...
        else:
            start_bin = random.randint(0, S_clean.shape[2] - 257)
        end_bin = start_bin + 257
        S_clean_frame = S_clean[:, :, start_bin:end_bin]
        S_lossy_frame = S_lossy[:, :, start_bin:end_bin]

        return S_lossy_frame, S_clean_frame  # complex output [1, 257, 257]      



if __name__ == "__main__":
    from utils.utils import vctk_parse
    import yaml
    import matplotlib.pyplot as plt

    with open(r'config.yaml') as file:
        par = yaml.load(file, Loader=yaml.FullLoader)
    par["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    # parse dataset
    train_list, val_list, _ = vctk_parse(par["VCTK_DIR"], par["VCTK_META"])

    # define VCTK dataset
    train_dataset = VCTK_dataset_raw(train_list, par, 'train')

    l, c = train_dataset[0]
    print(f'DATA shape: {l.shape} {c.shape}')

    S_clean_db = F.amplitude_to_DB(torch.abs(c), multiplier=20, amin=1e-6, db_multiplier=10)
    S_lossy_db = F.amplitude_to_DB(torch.abs(l), multiplier=20, amin=1e-6, db_multiplier=10)
    plt.subplot(1,2,1)
    plt.imshow(S_clean_db[0].detach().cpu(), origin='lower')
    plt.subplot(1,2,2)
    plt.imshow(S_lossy_db[0].detach().cpu(), origin='lower')
    plt.savefig('datasample.png')
        
    