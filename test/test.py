"""
Test functions to generate inpainted samples + PESQ, STOI, PLCMOSv2 and DNSMOS calculation
"""
import os, sys
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from pesq import pesq
from pystoi import stoi
from speechmos import plcmos, dnsmos
import librosa
import yaml
import torch
import torchaudio
from datetime import datetime
from utils.test_val_utils import FW_inpainter_VCTK, FW_inpainter_VCTK_custom, FW_inpainter_PLCC2022
from utils.utils import load_checkpoint, vctk_parse, file_parse
from networks.tcndenseunet import TCNDenseUNet



def test_VCTK(cli_args, par):
    
    test_type = 'ISTFT_stride160_plr20'
    

	# create output directories
    if not os.path.exists(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'CLEAN_VCTK')):
        os.makedirs(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'CLEAN_VCTK'))
    if not os.path.exists(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'LOSSY_VCTK')):
        os.makedirs(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'LOSSY_VCTK'))
    if not os.path.exists(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'INPAINTED_VCTK')):
        os.makedirs(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'INPAINTED_VCTK'))

    I = FW_inpainter_VCTK_custom(par, test_plr=args.test_plr)

    trained_gen = TCNDenseUNet().to(par["DEVICE"])
    trained_gen.eval()

    load_checkpoint(os.path.join(par["LOG_PATH"], cli_args.eid, par["MODEL_CHKPT_DIR"], 'chkpt_gen_e_' + str(cli_args.ckpt) + '.pth'),
                    trained_gen,
                    par["DEVICE"])

    _, _, test_list = vctk_parse(par["VCTK_DIR"], par["VCTK_META"])
    
    # test_list = test_list[0:10] # select only a subset of test files for debug

    print('Generate inpainted files...')
    start_tstamp = datetime.now()
    metrics_dict = []
    for f_idx, full_fname in enumerate(test_list, start=1):
        fname = os.path.split(full_fname)[1]
        print(f'{f_idx:3}/{len(test_list)} - Restoring {fname}...')

        # forward inpainting pass
        y_clean, y_lossy, y_restored = I.infer(full_fname, trained_gen, F2T_mode=cli_args.f2t_mode)
        y_c = np.array(y_clean[0])
        y_l = np.array(y_lossy[0])
        y_r = np.array(y_restored[0])
        try:
            P_clean_lossy = pesq(par["SAMPLE_RATE"], y_c, y_l, 'wb')
            P_clean_restored = pesq(par["SAMPLE_RATE"], y_c, y_r, 'wb')
            S_clean_lossy = stoi(y_c, y_l, par["SAMPLE_RATE"], extended=False)
            S_clean_restored = stoi(y_c, y_r, par["SAMPLE_RATE"], extended=False)
            restored_plcmos = plcmos.run(y_r, sr=par["SAMPLE_RATE"])
            PLCMOS = restored_plcmos["plcmos"]
            restored_dnsmos = dnsmos.run(y_r, sr=par["SAMPLE_RATE"])
            DNSMOS_ovrl = restored_dnsmos["ovrl_mos"]
            DNSMOS_sig = restored_dnsmos["sig_mos"]
            DNSMOS_bak = restored_dnsmos["bak_mos"]
            DNSMOS_p808 = restored_dnsmos["p808_mos"]
        except:
            print(f'an error occurred on file {fname}')

        metrics_dict.append({'filename': fname,
                             'pesq_clean_lossy_mean': P_clean_lossy,
                             'stoi_clean_lossy_mean': S_clean_lossy,
                             'pesq_restored_mean': P_clean_restored,
                             'stoi_restored_mean': S_clean_restored,
                             'plcmos_restored_mean': PLCMOS,
                             'dnsmos_ovrl_restored_mean': DNSMOS_ovrl,
                             'dnsmos_sig_restored_mean': DNSMOS_sig,
                             'dnsmos_bak_restored_mean': DNSMOS_bak,
                             'dnsmos_p808_restored_mean': DNSMOS_p808}),

        # save selected
        torchaudio.save(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'CLEAN_VCTK', fname), y_clean, par["SAMPLE_RATE"])
        torchaudio.save(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'LOSSY_VCTK', fname), y_lossy, par["SAMPLE_RATE"])
        torchaudio.save(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'INPAINTED_VCTK', fname), y_restored, par["SAMPLE_RATE"])
    
    df = pd.DataFrame.from_dict(metrics_dict)
    csv_path = os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'ALL_metrics_' + test_type + '.csv')
    df.to_csv (csv_path, index=False, header=True)
    print(f'{csv_path} saved')
    end_tstamp = datetime.now()
    print(f'Process started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Process finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')


def test_PLCChallenge2022(cli_args, par):

    clean_path = "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/"
    lossy_path = "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/lossy_signals/"
    restored_path = "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/restored_signals/bin2bin_cplx_exp061/"

    test_type = 'FGLA_stride1024_PLCC22'

	# create output directories
    if not os.path.exists(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'CLEAN_PLCC22')):
        os.makedirs(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'CLEAN_PLCC22'))
    if not os.path.exists(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'LOSSY_PLCC22')):
        os.makedirs(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'LOSSY_PLCC22'))
    if not os.path.exists(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'INPAINTED_PLCC22')):
        os.makedirs(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'INPAINTED_PLCC22'))

    I = FW_inpainter_PLCC2022(par, clean_path, lossy_path, restored_path)

    trained_gen = TCNDenseUNet().to(par["DEVICE"])
    trained_gen.eval()

    load_checkpoint(os.path.join(par["LOG_PATH"], cli_args.eid, par["MODEL_CHKPT_DIR"], 'chkpt_gen_e_' + str(cli_args.ckpt) + '.pth'),
                    trained_gen,
                    par["DEVICE"])

    test_list = file_parse(clean_path, "wav", return_fullpath=False)
    
    # test_list = test_list[0:10] # select only a subset of test files for debug

    print('Generate inpainted files...')
    start_tstamp = datetime.now()
    metrics_dict = []

    for f_idx, full_fname in enumerate(test_list, start=1):
        fname = os.path.split(full_fname)[1]
        print(f'{f_idx:3}/{len(test_list)} - Restoring {fname}...')

        y_clean, y_lossy, y_restored = I.infer(full_fname, trained_gen, F2T_mode=cli_args.f2t_mode)
        y_c = np.array(y_clean[0])
        y_l = np.array(y_lossy[0])
        y_r = np.array(y_restored[0])
        try:
            P_clean_lossy = pesq(par["SAMPLE_RATE"], y_c, y_l, 'wb')
            P_clean_restored = pesq(par["SAMPLE_RATE"], y_c, y_r, 'wb')
            S_clean_lossy = stoi(y_c, y_l, par["SAMPLE_RATE"], extended=False)
            S_clean_restored = stoi(y_c, y_r, par["SAMPLE_RATE"], extended=False)
            restored_plcmos = plcmos.run(y_r, sr=par["SAMPLE_RATE"])
            PLCMOS = restored_plcmos["plcmos"]
            restored_dnsmos = dnsmos.run(y_r, sr=par["SAMPLE_RATE"])
            DNSMOS_ovrl = restored_dnsmos["ovrl_mos"]
            DNSMOS_sig = restored_dnsmos["sig_mos"]
            DNSMOS_bak = restored_dnsmos["bak_mos"]
            DNSMOS_p808 = restored_dnsmos["p808_mos"]
        except:
            print(f'an error occurred on file {fname}')
            # pass

        metrics_dict.append({'filename': fname,
                             'pesq_clean_lossy': np.array(P_clean_lossy),
                             'stoi_clean_lossy': np.array(S_clean_lossy),
                             'pesq_restored': np.array(P_clean_restored),
                             'stoi_restored': np.array(S_clean_restored),
                             'plcmos_restored': np.array(PLCMOS),
                             'dnsmos_ovrl_restored': np.array(DNSMOS_ovrl),
                             'dnsmos_sig_restored': np.array(DNSMOS_sig),
                             'dnsmos_bak_restored': np.array(DNSMOS_bak),
                             'dnsmos_p808_restored': np.array(DNSMOS_p808)}),

        # save audio
        torchaudio.save(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'CLEAN_PLCC22', fname), y_clean, par["SAMPLE_RATE"])
        torchaudio.save(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'LOSSY_PLCC22', fname), y_lossy, par["SAMPLE_RATE"])
        torchaudio.save(os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'INPAINTED_PLCC22', fname), y_restored, par["SAMPLE_RATE"])
    
    df = pd.DataFrame.from_dict(metrics_dict)
    csv_path = os.path.join(par["LOG_PATH"], cli_args.eid, 'tests', test_type, 'ALL_metrics_' + test_type + '.csv')
    df.to_csv (csv_path, index=False, header=True)
    print(f'{csv_path} saved')
    end_tstamp = datetime.now()
    print(f'Process started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Process finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')


def evaluate_metrics_only(par):

    clean_path = "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/"
    lossy_path = "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/lossy_signals/"
    restored_path = "/mnt/sda1/home/caironi/CPLX_BIN2BIN/logs/050/tests/ISTFT_stride1024_PLCC22/INPAINTED_PLCC22"

    test_list = file_parse(clean_path, "wav", return_fullpath=False)
    # test_list = test_list[0:10] # select only a subset of test files for debug

    metrics_dict = []

    for full_fname in tqdm(test_list):
        fname = os.path.split(full_fname)[1]

        y_clean, _ = torchaudio.load(os.path.join(clean_path, fname))
        y_lossy, _ = torchaudio.load(os.path.join(lossy_path, fname))
        y_restored, _ = torchaudio.load(os.path.join(restored_path, fname))

        y_c = np.array(y_clean[0])
        y_l = np.array(y_lossy[0])
        y_r = np.array(y_restored[0])

        try:
            P_clean_lossy = pesq(par["SAMPLE_RATE"], y_c, y_l, 'wb')
            P_clean_restored = pesq(par["SAMPLE_RATE"], y_c, y_r, 'wb')
            S_clean_lossy = stoi(y_c, y_l, par["SAMPLE_RATE"], extended=False)
            S_clean_restored = stoi(y_c, y_r, par["SAMPLE_RATE"], extended=False)
            restored_plcmos = plcmos.run(y_r, sr=par["SAMPLE_RATE"])
            PLCMOS = restored_plcmos["plcmos"]
            restored_dnsmos = dnsmos.run(y_r, sr=par["SAMPLE_RATE"])
            DNSMOS_ovrl = restored_dnsmos["ovrl_mos"]
            DNSMOS_sig = restored_dnsmos["sig_mos"]
            DNSMOS_bak = restored_dnsmos["bak_mos"]
            DNSMOS_p808 = restored_dnsmos["p808_mos"]
        except:
            print(f'an error occurred on file {fname}')

        metrics_dict.append({'filename': fname,
                             'pesq_clean_lossy': np.array(P_clean_lossy),
                             'stoi_clean_lossy': np.array(S_clean_lossy),
                             'pesq_restored': np.array(P_clean_restored),
                             'stoi_restored': np.array(S_clean_restored),
                             'plcmos_restored': np.array(PLCMOS),
                             'dnsmos_ovrl_restored': np.array(DNSMOS_ovrl),
                             'dnsmos_sig_restored': np.array(DNSMOS_sig),
                             'dnsmos_bak_restored': np.array(DNSMOS_bak),
                             'dnsmos_p808_restored': np.array(DNSMOS_p808)}),
    
    df = pd.DataFrame.from_dict(metrics_dict)
    df.to_csv ("metriche_cplxbin2bin.csv", index=False, header=True)
    print('csv saved')


def evaluate_MOS_only(par):
 
    clean_path = "/mnt/sda1/home/caironi/CPLX_BIN2BIN/logs/061/tests/ISTFT_stride20_plr50/CLEAN_VCTK"
    lossy_path = "/mnt/sda1/home/caironi/CPLX_BIN2BIN/logs/061/tests/ISTFT_stride20_plr50/LOSSY_VCTK"
    restored_path = "/mnt/sda1/home/caironi/CPLX_BIN2BIN/logs/050/tests/ISTFT_stride1024_PLCC22/INPAINTED_PLCC22"

    test_list = file_parse(clean_path, "wav", return_fullpath=False)
    # test_list = test_list[0:10] # select only a subset of test files for debug

    DNSMOS_clean = np.zeros(4) # placeholder for [ovrl, sig, bak, p808]
    DNSMOS_lossy = np.zeros(4)
    DNSMOS_restored = np.zeros(4)
    PCLMOS_clean = 0.0
    PCLMOS_lossy = 0.0
    PCLMOS_restored = 0.0
    

    for full_fname in tqdm(test_list):
        fname = os.path.split(full_fname)[1]

        y_clean, _ = torchaudio.load(os.path.join(clean_path, fname))
        y_lossy, _ = torchaudio.load(os.path.join(lossy_path, fname))
        y_restored, _ = torchaudio.load(os.path.join(restored_path, fname))

        y_c = np.array(y_clean[0])
        y_l = np.array(y_lossy[0])
        y_r = np.array(y_restored[0])

        try:
            clean_dnsmos = dnsmos.run(y_c, sr=par["SAMPLE_RATE"])
            DNSMOS_clean[0] += clean_dnsmos["ovrl_mos"]
            DNSMOS_clean[1] += clean_dnsmos["sig_mos"]
            DNSMOS_clean[2] += clean_dnsmos["bak_mos"]
            DNSMOS_clean[3] += clean_dnsmos["p808_mos"]
            PCLMOS_clean += plcmos.run(y_c, sr=par["SAMPLE_RATE"])["plcmos"]

            lossy_dnsmos = dnsmos.run(y_l, sr=par["SAMPLE_RATE"])
            DNSMOS_lossy[0] += lossy_dnsmos["ovrl_mos"]
            DNSMOS_lossy[1] += lossy_dnsmos["sig_mos"]
            DNSMOS_lossy[2] += lossy_dnsmos["bak_mos"]
            DNSMOS_lossy[3] += lossy_dnsmos["p808_mos"]
            PCLMOS_lossy += plcmos.run(y_l, sr=par["SAMPLE_RATE"])["plcmos"]

            restored_dnsmos = dnsmos.run(y_r, sr=par["SAMPLE_RATE"])
            DNSMOS_restored[0] += restored_dnsmos["ovrl_mos"]
            DNSMOS_restored[1] += restored_dnsmos["sig_mos"]
            DNSMOS_restored[2] += restored_dnsmos["bak_mos"]
            DNSMOS_restored[3] += restored_dnsmos["p808_mos"]
            PCLMOS_restored += plcmos.run(y_r, sr=par["SAMPLE_RATE"])["plcmos"]

        except:
            print(f'an error occurred on file {fname}')
        
    print(f'DNSMOS CLEAN:    {DNSMOS_clean[0]/len(test_list):.5f} (ovrl)    {DNSMOS_clean[1]/len(test_list):.5f} (sig)    {DNSMOS_clean[2]/len(test_list):.5f} (bak)    {DNSMOS_clean[3]/len(test_list):.5f} (p808)')
    print(f'PLCMOSv2 CLEAN:  {PCLMOS_clean/len(test_list):.5f}')
    print(f'DNSMOS LOSSY:    {DNSMOS_lossy[0]/len(test_list):.5f} (ovrl)    {DNSMOS_lossy[1]/len(test_list):.5f} (sig)    {DNSMOS_lossy[2]/len(test_list):.5f} (bak)    {DNSMOS_lossy[3]/len(test_list):.5f} (p808)')
    print(f'PLCMOSv2 LOSSY:  {PCLMOS_lossy/len(test_list):.5f}')
    print(f'DNSMOS RESTORED: {DNSMOS_restored[0]/len(test_list):.5f} (ovrl)    {DNSMOS_restored[1]/len(test_list):.5f} (sig)    {DNSMOS_restored[2]/len(test_list):.5f} (bak)    {DNSMOS_restored[3]/len(test_list):.5f} (p808)')
    print(f'PLCMOSv2 RESTORED:  {PCLMOS_restored/len(test_list):.5f}')
    print('done')



if __name__ == "__main__":
    """    
    1 - change test_type
    2 - run $python test.py -eid 061 -test_plr 0.1 -ckpt BEST -f2t_mode ISTFT
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', required = False, help = 'Experiment ID')
    parser.add_argument('-test_plr', type=float, required = False, help = 'Loss rate for test (override config param)')
    parser.add_argument('-ckpt', required = False, help = 'Checkpoint epoch to load, BEST_P, BEST_S or BEST_PLCMOS')
    parser.add_argument('-f2t_mode', required = False, help = 'FGLA or ISTFT')
    args = parser.parse_args()

    # parse parameters from file
    with open(r'config.yaml') as file:
        par = yaml.load(file, Loader=yaml.FullLoader)
    par["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    test_VCTK(args, par)
    # test_PLCChallenge2022(args, par)
    # evaluate_metrics_only(par)
    # evaluate_MOS_only(par)