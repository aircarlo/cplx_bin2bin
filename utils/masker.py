from tqdm import tqdm
import yaml
import os, sys
import torch
import torchaudio
import torchaudio.transforms as T
from utils import vctk_parse
import numpy as np


class Masker(object):
    """
    Generate gap masks and save them to npy files
    """
    def __init__(self, wav_list, dest_dir, params):
        self.wav_list = wav_list
        self.dest_dir = dest_dir
        self.par = params

        self.resampler = T.Resample(self.par["VCTK_SAMPLE_RATE"],
                                    self.par["SAMPLE_RATE"],
                                    resampling_method="kaiser_window",
                                    dtype=torch.float32)

    def make_masks(self):
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        else:
            raise Exception(f'Directory {self.dest_dir} already exists!')
        for idx in tqdm(range(len(self.wav_list))):
            # load the clean waveform
            clean_waveform, _ = torchaudio.load(self.wav_list[idx])
            clean_waveform = self.resampler(clean_waveform)  # downsample from 48kHz to 16kHz

            # trim to 20ms
            len_samples = clean_waveform.shape[1]
            clean_waveform = clean_waveform[:, 0:(len_samples // 320 * 320)]

            len_frames = len_samples // 320  # 320 samples = 1 frame = 20ms @ 16kHz
            frame_mask = np.random.choice([0, 1], size=(len_frames,), p=[self.par["LOSSY_RATE"], 1-self.par["LOSSY_RATE"]])
            mask_fname = (self.wav_list[idx][:-3] + 'npy').split('/')[-1]
            np.save(os.path.join(self.dest_dir, mask_fname), frame_mask)

    def get_mask_stats():
        # TODO: to be implemented
        pass
        
        

if __name__ == "__main__":
    if sys.argv[1] not in ['validate', 'val', 'test']:
        raise Exception('Please specify "val" or "test"')

    with open(r'config.yaml') as file:
        par = yaml.load(file, Loader=yaml.FullLoader)
    _, val_list, test_list = vctk_parse(par["VCTK_DIR"], par["VCTK_META"])
    if sys.argv[1] in ['val', 'validate']:
        mask_generator = Masker(val_list, par["VAL_MASK_PATH"], par)
    if sys.argv[1] in ['test']:
        mask_generator = Masker(test_list, par["TEST_MASK_PATH"], par)
    mask_generator.make_masks()