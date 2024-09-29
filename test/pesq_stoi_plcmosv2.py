"""
Test function to calculate PESQ, STOI and PLCMOSv2 on a set of files
"""

import os
import yaml
import argparse
from tqdm import tqdm
import librosa
from pesq import pesq
from pystoi import stoi
from speechmos import plcmos
from utils.utils import wav_meta_parse


def start_test(args):
    """
    $python pesq_stoi_plcmos.py --clean_dir <CLEAN FILES PATH> --lossy_dir <LOSSY FILES PATH> --restored_dir <INPAINTED FILES PATH>
    python pesq_stoi_plcmos.py --clean_dir logs/051/tests/ISTFT_stride20_plr5/CLEAN_VCTK --lossy_dir logs/051/tests/ISTFT_stride20_plr5/LOSSY_VCTK --restored_dir logs/051/tests/ISTFT_stride20_plr5/INPAINTED_VCTK
    """
    filelist, _ = wav_meta_parse(args.clean_dir)

    P_clean_lossy = []
    P_clean_restored = []
    S_clean_lossy = []
    S_clean_restored = []
    PLCMOS_restored = []

    for fname in tqdm(filelist):
        try:
            y_clean, _ = librosa.load(os.path.join(args.clean_dir, fname), sr=par["SAMPLE_RATE"])
            y_lossy, _ = librosa.load(os.path.join(args.lossy_dir, fname), sr=par["SAMPLE_RATE"])
            y_restored, _ = librosa.load(os.path.join(args.restored_dir, fname), sr=par["SAMPLE_RATE"])

            P_clean_lossy.append(pesq(16000, y_clean, y_lossy, 'wb'))
            P_clean_restored.append(pesq(16000, y_clean, y_restored, 'wb'))
            S_clean_lossy.append(stoi(y_clean, y_lossy, par["SAMPLE_RATE"], extended=False))
            S_clean_restored.append(stoi(y_clean, y_restored, par["SAMPLE_RATE"], extended=False))
            PLCMOS = plcmos.run(y_restored, sr=par["SAMPLE_RATE"])
            PLCMOS_restored.append(PLCMOS["plcmos"]) 

        except:
            print(f'an error occurred on file {fname}')

    print(f'PESQ (LOSSY-CLEAN): {sum(P_clean_lossy)/len(P_clean_lossy)}')
    print(f'STOI (LOSSY-CLEAN): {sum(S_clean_lossy)/len(S_clean_lossy)}')
    print(f'PESQ (RESTORED-CLEAN): {sum(P_clean_restored)/len(P_clean_restored)}')
    print(f'STOI (RESTORED-CLEAN): {sum(S_clean_restored)/len(S_clean_restored)}')
    print(f'PLCMOS (RESTORED): {sum(PLCMOS_restored)/len(PLCMOS_restored)}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--clean_dir", default='.', help='Directory containing clean files')
    parser.add_argument('-l', "--lossy_dir", default='.', help='Directory containing lossy files')
    parser.add_argument('-r', "--restored_dir", default='.', help='Directory containing restored (inpainted) files')
    parser.add_argument('-o', "--out_csv", default=None, help='Path to the csv file ') # TODO: implement csv output
    args = parser.parse_args()

    # parse parameters from file
    with open(r'config.yaml') as file:
        par = yaml.load(file, Loader=yaml.FullLoader)

    start_test(args)