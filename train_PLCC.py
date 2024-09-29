"""
Train script for PLCC_2022 dataset only
"""
import os, sys
import yaml
from tqdm import tqdm
from datetime import datetime
import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
# evaluation metrics
from pesq import pesq
from pystoi import stoi
from speechmos import plcmos
# losses
import auraloss
from asteroid.losses import SingleSrcPMSQE
# other modules
from utils.test_val_utils import FW_inpainter_PLCC2022
from utils.utils import save_checkpoint, save_gen_specs, vctk_parse, count_parameters, file_parse
from speech_dataset import VCTK_dataset_raw, PLCDataset_raw
from networks.discriminator_model import Discriminator
from networks.tcndenseunet import TCNDenseUNet
from utils.losses import WrappedPhaseLoss


# torch.autograd.set_detect_anomaly(True)

a2db = T.AmplitudeToDB()
def raw_to_magnitude(x, p=1): # p=1 for energy, p=2 for power spectrograms
    return a2db(torch.abs(x)**p)

def train_fn(disc, gen, loader, opt_disc, opt_gen, loss, lr_schedulers):

    loop = tqdm(loader, leave=True)
    G_MR_loss_epoch = 0.0
    
    disc.train()
    gen.train()

    for idx, (x, y) in enumerate(loop):
        
        # Train the discriminator
        opt_disc.zero_grad()
        y_fake = gen(x)
        D_real = disc(raw_to_magnitude(x), raw_to_magnitude(y))         # discriminator is fed with lossy and clean amplitude/power spectrograms in dB
        D_real_loss = loss[0](D_real, torch.ones_like(D_real))          # 1 = "real"
        D_fake = disc(raw_to_magnitude(x), raw_to_magnitude(y_fake.detach()))
        D_fake_loss = loss[0](D_fake, torch.zeros_like(D_fake))         # 0 = "fake"
        D_loss = (D_real_loss + D_fake_loss) / 2

        if not torch.isnan(D_loss):
            if idx % par["N_CRITICS"] == 0: # update discriminator every n_critics iterations
                D_loss.backward()
                # torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)  # gradient clipping, deprecated
                opt_disc.step()

        # Train the generator
        opt_gen.zero_grad()
        D_fake = disc(raw_to_magnitude(x), raw_to_magnitude(y_fake))
        G_MSE_loss = loss[0](D_fake, torch.ones_like(D_fake))           # adversarial loss
        # multiresolution loss
        y_fake_t = torch.istft(y_fake.squeeze(1), n_fft=par["FFT_SIZE"])
        y_t = torch.istft(y.squeeze(1), n_fft=par["FFT_SIZE"])
        G_MR_loss = loss[1](y_fake_t.unsqueeze(1), y_t.unsqueeze(1))
        # perceptual loss
        PMSQE_loss = torch.mean(loss[2](torch.abs(y_fake)[:,0,:,:], torch.abs(y)[:,0,:,:])) # * par["LAMBDA"]
        PHA_loss = loss[3](y, y_fake)

        G_loss = G_MSE_loss + G_MR_loss + 0.1*PHA_loss + PMSQE_loss
        
        G_MR_loss_epoch += G_MR_loss.item()

        if not torch.isnan(G_loss):
            # update generator at every iteration
            G_loss.backward()
            # torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0) # gradient clipping, deprecated
            opt_gen.step()
        
        #### early stop for debug
        if idx==50:
            break
        ####

    lr_schedulers[0].step()
    lr_schedulers[1].step()
    lr_values = (opt_disc.param_groups[0]["lr"], opt_gen.param_groups[0]["lr"])

    return G_MR_loss_epoch, lr_values



@torch.no_grad()
def val_metrics_fn(gen, filelist, inpainter):
    """
    Validation function operating at file level.
    It computes PESQ, STOI and PLCMOSv2 using the same inpainting procedure of test
    """
    P, S, PLCM = [], [], []   # PESQ, STOI and PLCMOS buffers
    gen.eval()
    for idx in tqdm(range(len(filelist))):
        fname = filelist[idx]
        y_clean, _, y_inpainted = inpainter.infer(fname, gen, F2T_mode='ISTFT')
        y_clean = torch.squeeze(y_clean)
        y_inpainted = torch.squeeze(y_inpainted)
        try:
            P.append(pesq(par["SAMPLE_RATE"], y_clean.numpy(), y_inpainted.numpy(), 'wb'))
        except:
            pass
        try:
            S.append(stoi(y_clean.numpy(), y_inpainted.numpy(), par["SAMPLE_RATE"], extended=False))
        except:
            pass
        try:
            plcm = plcmos.run(y_inpainted.numpy(), sr=par["SAMPLE_RATE"])
            PLCM.append(plcm['plcmos'])
        except:
            pass

    P = sum(P) / len(P)
    S = sum(S) / len(S)
    PLCM = sum(PLCM) / len(PLCM)

    return P, S, PLCM


def main():

    # create log folder
    if not os.path.exists(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])):
        os.makedirs(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"]))
        os.makedirs(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["IMG_EVAL_DIR"]))
        os.makedirs(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"]))
    else:
        raise Exception(f'Log folder {os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])} already exists!')
    
    now_tstamp = datetime.now()

    # dump yaml param file to txt
    log_info = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], "params.txt")
    with open(log_info, 'w') as f:
        f.write(f'Experiment {par["EXPERIMENT_ID"]} started at {now_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}\n\n')
        for k in par.keys():
            f.write(f'{k}: {par[k]}\n')

    # initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"]))

    # define networks
    disc = Discriminator(in_channels=1).to(par["DEVICE"])
    gen = TCNDenseUNet(**par["TDU_PARAM"]).to(par["DEVICE"])

    # print(f'GENERATOR PARAMS: {count_parameters(gen)}')
    # print(f'DISCRIMINATOR PARAMS: {count_parameters(disc)}')
    # sys.exit()

    # optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=par["D_LEARNING_RATE"], betas=(0.9, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=par["G_LEARNING_RATE"], betas=(0.9, 0.999))

    # learning rate schedulers
    opt_disc_scheduler = CosineAnnealingLR(opt_disc,
                                           T_max = par["NUM_EPOCHS"], # Maximum number of iterations.
                                           eta_min = 0.001) # Minimum learning rate.
    opt_gen_scheduler = CosineAnnealingLR(opt_gen,
                                          T_max = par["NUM_EPOCHS"], # Maximum number of iterations.
                                          eta_min = 0.001) # Minimum learning rate.

    # list of available loss fn
    adv_loss = nn.MSELoss()             # LSGAN -> MSELoss
    mrloss = auraloss.freq.MultiResolutionSTFTLoss( # for more parameters see https://github.com/csteinmetz1/auraloss
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sc=par['MR_SC'],
        w_log_mag=par['MR_LOG'],
        w_lin_mag=par['MR_LIN'],
        w_phs=par['MR_PHS'],
        sample_rate=par["SAMPLE_RATE"],
        scale=None,
        n_bins=None,
        scale_invariance=False,
        device=par['DEVICE']
    )
    pmsqe_loss = SingleSrcPMSQE().to(par['DEVICE'])
    phase_loss = WrappedPhaseLoss().to(par['DEVICE'])

    train_list2022 = file_parse("/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/train/clean_signals/",
                                "wav",
                                return_fullpath=True)
    train_list2024 = file_parse("/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2024/validation_data_v2/val_clean/",
                                "wav",
                                return_fullpath=True)
    train_list = train_list2022 + train_list2024

    val_list = ["/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/6373.wav",
                "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/309854.wav",
                "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/418901.wav",
                "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/99225.wav",
                "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/224176.wav",
                "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/434938.wav"]
    
    # define dataset and dataloader
    train_dataset = PLCDataset_raw(train_list, par)
    val_dataset = PLCDataset_raw(val_list, par)
    
    val_I = FW_inpainter_PLCC2022(par,
                              "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/reference_signals/",
                              "/mnt/sda1/home/caironi/DATA/IS_PLCchallenge_2022/blind/lossy_signals/",
                              None)

    train_loader = DataLoader(train_dataset,
                              batch_size=par["BATCH_SIZE"],
                              shuffle=True,
                              num_workers=par["NUM_WORKERS"])

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=par["NUM_WORKERS"])
    
    visual_proof_sample = val_dataset[0] 

    print(f'Start training experiment {par["EXPERIMENT_ID"]}')
    start_tstamp = datetime.now()

    # buffers
    val_metrics_result = [0]*3      # initialize with zeros because tensorboard requires values even if validation_file is skipped
    best_P, best_S, best_PLCMOS = 0, 0, 0  # for checkpoint

    # start
    for epoch in range(par["NUM_EPOCHS"]):

        print(f'\nEPOCH: {epoch+1}/{par["NUM_EPOCHS"]} - Train...')

        train_result = train_fn(disc,
                                gen,
                                train_loader,
                                opt_disc,
                                opt_gen,
                                [adv_loss, mrloss, pmsqe_loss, phase_loss],
                                lr_schedulers = [opt_disc_scheduler, opt_gen_scheduler])

        print(f'File-level validation...')
        val_metrics_result = val_metrics_fn(gen,
                                            val_list,
                                            val_I)

        # Tensorboard logs
        writer.add_scalar('G_MR_LOSS',      (train_result[0]/len(train_loader)), epoch) # Generator MultiResolution loss
        writer.add_scalar('PESQ_VAL',       (val_metrics_result[0]), epoch)
        writer.add_scalar('STOI_VAL',       (val_metrics_result[1]), epoch)
        writer.add_scalar('PLCMOS_VAL',     (val_metrics_result[2]), epoch)
        writer.flush()


        # checkpoint at fixed steps (not used)
        if epoch+1 in par["SAVE_MODEL_AT"]:
            save_checkpoint(gen, opt_gen, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_gen_e_' + str(epoch) + '.pth'))
            # do not save discriminator                                                                                                                                                                          
            # save_checkpoint(disc, opt_disc, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_disc_e_' + str(epoch) + '.pth'))

        # checkpoint of best model according to PESQ
        current_P = val_metrics_result[0]
        if current_P > best_P:
            save_checkpoint(gen, opt_gen, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_gen_e_BEST_P.pth'))
            save_checkpoint(disc, opt_disc, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_disc_e_BEST_P.pth'))
            best_P = current_P
            best_P_checkpoint = epoch
            print('> best PESQ checkpoint saved')

        # checkpoint of best model according to STOI
        current_S = val_metrics_result[1]
        if current_S > best_S:
            save_checkpoint(gen, opt_gen, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_gen_e_BEST_S.pth'))
            save_checkpoint(disc, opt_disc, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_disc_e_BEST_S.pth'))
            best_S = current_S
            best_S_checkpoint = epoch
            print('> best STOI checkpoint saved')

        # checkpoint of best model according to PLCMOS
        current_PLCMOS = val_metrics_result[2]
        if current_PLCMOS > best_PLCMOS:
            save_checkpoint(gen, opt_gen, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_gen_e_BEST_PLCMOS.pth'))
            save_checkpoint(disc, opt_disc, filepath = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], par["MODEL_CHKPT_DIR"], 'chkpt_disc_e_BEST_PLCMOS.pth'))
            best_PLCMOS = current_PLCMOS
            best_PLCMOS_checkpoint = epoch
            print('> best PLCMOS checkpoint saved')

        if epoch+1 in par["SAVE_IMG_AT"]:
            save_gen_specs(gen, visual_proof_sample, epoch, par)

    end_tstamp = datetime.now()
    writer.close()
    print('DONE')
    print(f'Best PESQ at epoch {best_P_checkpoint}')
    print(f'Best STOI at epoch {best_S_checkpoint}')
    print(f'Best PLCMOS at epoch {best_PLCMOS_checkpoint}')
    print(f'Training started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Training finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')


if __name__ == "__main__":
    # parse parameters
    with open(r'config.yaml') as file:
        par = yaml.load(file, Loader=yaml.FullLoader)
    par["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    if par["SAVE_IMG_AT"]=='all':
        par["SAVE_IMG_AT"] = list(range(1,par["NUM_EPOCHS"]+1))
    main()