import os
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import whisper
import torchaudio
from tqdm import tqdm
import jiwer
from whisper.normalizers.english import EnglishTextNormalizer
from utils.utils import file_parse, file_transcription_parse


class PLCDataset_ASR(Dataset):
    """
    Custom dataset class to wrap PLCDataset files with the corresponding transcription.
    $python wer.py --test_dir /home/aironi/GAN/pix2pix_speech/logs/033/RESTORED --csv_file /home/aironi/dataset/PLC_challenge_data/test/transcriptions.csv
    """
    def __init__(self, root_dir, transcript, device):
        self.root_dir = root_dir
        self.transcript = transcript
        self.device = device
        # parse data
        self.wav_file_list = file_transcription_parse(self.root_dir, self.transcript)
        self.meta = pd.read_csv(transcript)

    def __len__(self):
        return len(self.wav_file_list)

    def __getitem__(self, idx):
        wav_id = self.wav_file_list[idx][:-4]
        waveform, sample_rate = torchaudio.load(os.path.join(self.root_dir, self.wav_file_list[idx]))
        transcript = self.meta.loc[self.meta['id'] == int(wav_id)].transcription.values[0]
        assert sample_rate == 16000
        waveform = whisper.pad_or_trim(waveform.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(waveform)
        
        return mel, transcript


class VCTK_dataset_ASR(Dataset):
    """
    Custom dataset class to wrap VCTK files with the corresponding transcription.
    Transcriptions are included in VCTK_TXT path
    """
    def __init__(self, params):
        self.params = params
        self.wav_path = cli_args.test_dir
        # parse data
        # _, _, self.wav_list = vctk_parse(self.params["VCTK_DIR"], self.params["VCTK_META"])
        self.wav_list = file_parse(root_dir = self.wav_path,
                                   ext = 'wav',
                                   return_fullpath=True)
        # self.resampler = T.Resample(48000, 16000)

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.wav_list[idx])
        # waveform = self.resampler(waveform)
        # txt_file = os.path.join(self.params["VCTK_TXT"], self.wav_list[idx].split('/')[-2], self.wav_list[idx].split('/')[-1][:-4] + '.txt')
        txt_file = os.path.join(self.params["VCTK_TXT"], self.wav_list[idx].split('/')[-1][0:4], self.wav_list[idx].split('/')[-1][:-4] + '.txt')
        with open(txt_file) as f:
            transcript = f.readlines()[0]
        waveform = whisper.pad_or_trim(waveform.flatten()).to(self.params["DEVICE"])
        mel = whisper.log_mel_spectrogram(waveform)
        
        return mel, transcript


def main(cli_args, params):

    dataset = VCTK_dataset_ASR(params) # VCTK dataset
    # dataset = PLCDataset_ASR(cli_args.test_dir, cli_args.csv_file, params["DEVICE"]) # PLCChallenge dataset

    loader = DataLoader(dataset, batch_size=16)
    if cli_args.model == 'large':
        print('Warning, the selected model requires high VRAM ')

    print('Load ASR model...')
    model = whisper.load_model(cli_args.model)
    print(f"{'Multilingual' if model.is_multilingual else 'English-only'} {cli_args.model} model loaded ({sum(np.prod(p.shape) for p in model.parameters())} parameters)\n")

    # predict without timestamps for short-form transcription
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    hypotheses = []
    references = []

    print('Evaluate...')
    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    if cli_args.txtnorm == 'whisper':
        normalizer = EnglishTextNormalizer()
        data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
        data["reference_clean"] = [normalizer(text) for text in data["reference"]]
        # wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
        wm = jiwer.compute_measures(list(data["reference_clean"]), list(data["hypothesis_clean"]))

    elif cli_args.txtnorm == 'jiwer':
        transformation = jiwer.Compose([jiwer.ToLowerCase(),
                                        jiwer.RemoveWhiteSpace(replace_by_space=True),
                                        jiwer.RemoveMultipleSpaces(),
                                        jiwer.RemovePunctuation(),
                                        jiwer.Strip(),
                                        jiwer.ExpandCommonEnglishContractions(),
                                        jiwer.ReduceToListOfListOfWords(word_delimiter=" "), # this normalization must be in the last place
                                        ])
        # https://github.com/jitsi/jiwer see for more normalizations

        wer = jiwer.wer(list(data["reference"]), 
                        list(data["hypothesis"]), 
                        truth_transform=transformation, 
                        hypothesis_transform=transformation)


    # print(f"WER: {wer * 100:.2f} % - ACC: {(1-wer) * 100:.2f}")
    print(f"WER: {wm['wer'] * 100:.2f} % - INSERTIONS: {wm['insertions']} - SUBSTITUTIONS: {wm['substitutions']} - DELETIONS: {wm['deletions']}")


if __name__ == "__main__":
    """
    $python wer.py --test_dir /mnt/sda1/home/caironi/CPLX_BIN2BIN/logs/051/tests/ISTFT_stride20_plr5/CLEAN_VCTK
    $python wer.py --test_dir /mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/lossy_signals/ --csv_file /mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/transcriptions.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--test_dir", default='.', help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-m', "--model", choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], default='medium.en', help='Pretrained ASR model. Those with .en are english-only')
    parser.add_argument('-b', "--b_size", default=16, help='Batch size')
    parser.add_argument('-n', "--txtnorm", default='whisper', help='Text normalization engine: whisper or jiwer')
    parser.add_argument('-c', "--csv_file", default=None, help='Path to the csv file containing the transcriptions (only for PLCChallenge dataset)')

    cli_args = parser.parse_args()

    with open(r'config.yaml') as file:
        par = yaml.load(file, Loader=yaml.FullLoader)
    par["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(cli_args, par)