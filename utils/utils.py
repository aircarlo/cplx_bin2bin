import os
import torch
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import pandas as pd



def PLC22_parse(clean_dir, lossy_dir):
    """
    parse PLC 2022 dataset (with gap masks on txt files)
    """
    clean_list = []
    lossy_list = []
    mask_list = []
    print(f'parse wav files in {clean_dir}... ')
    for root, _, files in os.walk(clean_dir, topdown=False):
        for name in files:
            if os.path.isfile(os.path.join(root, name)):
                if name[-3:] == 'wav':
                    clean_list.append(name)
    print(f'parse wav files in {lossy_dir}... ')
    for root, _, files in os.walk(lossy_dir, topdown=False):
        for name in files:
            if os.path.isfile(os.path.join(root, name)):
                if name[-3:] == 'wav':
                    lossy_list.append(name)
    print(f'parse mask files in {lossy_dir}... ')
    for root, _, files in os.walk(lossy_dir, topdown=False):
        for name in files:
            if os.path.isfile(os.path.join(root, name)):
                if name[-3:] == 'txt':
                    mask_list.append(name)
    clean_bare = [i[:-4] for i in clean_list]
    lossy_bare = [i[:-4] for i in lossy_list]
    mask_bare = [i[:-12] for i in mask_list]  # mask file name is like '123_is_lost.txt'
    # check if every clean_wav has the corresponding lossy_wav and meta mask
    if len(clean_list)==len(lossy_list)==len(mask_list) and set(clean_bare)==set(lossy_bare)==set(mask_bare):
        print('check pass!\n')
    else:
        raise ValueError('check fail!')

    return clean_bare # return only file name without extension



def file_parse(root_dir, ext, return_fullpath=True):
    """
    parse files with generic extensions
    """
    file_list = []
    print(f'parse {root_dir}... ', end='')
    for root, _, files in os.walk(root_dir, topdown=False):
        for name in files:
            if os.path.isfile(os.path.join(root, name)):
                if name[-3:] == ext:
                    if return_fullpath:
                        file_list.append(os.path.join(root, name))
                    else:
                        file_list.append(name)
    print(f'found {len(file_list)} {ext} files')
    return file_list


def file_transcription_parse(data_dir, transcript):
    """
    lists wav files in dataset that have a valid transcription
    """
    wav_file_list = []
    print(f'parse {data_dir}...')
    meta_text = pd.read_csv(transcript)
    for fname in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, fname)):
            if fname[-3:] == 'wav':
                if meta_text.loc[meta_text['id'] == int(fname[:-4])].exclude.values[0] == False:
                    wav_file_list.append(fname)
    print(f'found {len(wav_file_list)} wav files with corresponding transcriptions\n')
    return wav_file_list



def vctk_parse(data_path, meta_file):
    """
    parse VCTK corpus
    """
    import csv
    import random
    train_ids = []
    val_ids = []
    test_ids = []
    train_wav = []
    val_wav = []
    test_wav = []

    print(f'parse {data_path}... ')

    
    with open(meta_file, "r", encoding="utf8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        next(tsv_reader) # skip the header row

        # Train-Val split based on meta file
        for row in tsv_reader:
            id, split, *_ = row[0].split()
            if split == 'TRAIN':
                train_ids.append(id)
            if split == 'VAL':
                val_ids.append(id)
            if split == 'TEST':
                test_ids.append(id)
        
        # # Train-Val split randomly
        # random.seed(10) # for reproducibility
        # ids = []
        # for row in tsv_reader:
        #     id, *_ = row[0].split()
        #     ids.append(id)
        # test_val_ids = random.sample(ids, 9)
        # ids = set(ids)-set(test_val_ids)
        # train_ids = set(ids)-set(test_val_ids)
        # test_ids = test_val_ids[0:5]
        # val_ids = test_val_ids[5:9]
    
    for root, _, files in os.walk(data_path, topdown=False):
        for name in files:
            if os.path.isfile(os.path.join(root, name)):
                if root[-3:] in train_ids and  name[-3:] == 'wav':
                    train_wav.append(os.path.join(root, name))
                elif root[-3:] in val_ids and  name[-3:] == 'wav':
                    val_wav.append(os.path.join(root, name))
                elif root[-3:] in test_ids and  name[-3:] == 'wav':
                    test_wav.append(os.path.join(root, name))

    print(f'found {len(train_wav)} wav files for train')
    print(f'found {len(val_wav)} wav files for validation')
    print(f'found {len(test_wav)} wav files for test')

    return train_wav, val_wav, test_wav



def get_max_gap(mask_ctxt):
    """
    look for the widest gap inside a mask
    """
    w = 0
    w_list = []
    for mask_s in mask_ctxt:
        if mask_s == 1:
            w +=1
        else:
            w=0
        w_list.append(w)
    return max(w_list)



def save_gen_specs(gen, sample, epoch, params):
    """
    save generator output as image
    """
    folder = os.path.join(params["LOG_PATH"], params["EXPERIMENT_ID"], params["IMG_EVAL_DIR"])
    x, y = sample
    x = x.unsqueeze(0).to(params["DEVICE"])
    y = y.unsqueeze(0).to(params["DEVICE"])

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake_db = F.amplitude_to_DB(torch.abs(y_fake)**2, multiplier=10, amin=1e-6, db_multiplier=10)
        plt.imshow(y_fake_db[0][0].detach().cpu().numpy(), origin='lower')
        plt.savefig(folder + f"/y_gen_{epoch}.png")
        plt.close()

        if epoch == 0: # at epoch 0 save also input (with gap) and target (clean)
            x_db = F.amplitude_to_DB(torch.abs(x)**2, multiplier=10, amin=1e-6, db_multiplier=10)
            plt.imshow(x_db[0][0].detach().cpu().numpy(), origin='lower')
            plt.savefig(folder + "/input.png")
            plt.close()
            y_db = F.amplitude_to_DB(torch.abs(y)**2, multiplier=10, amin=1e-6, db_multiplier=10)
            plt.imshow(y_db[0][0].detach().cpu().numpy(), origin='lower')
            plt.savefig(folder + "/target.png")
            plt.close()

    print('>> examples saved')



def save_checkpoint(model, optimizer, filepath):
    # print(">> checkpoint saved")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(checkpoint_file, model, device, optimizer=None, lr=None):
    print(f'>> loading checkpoint {checkpoint_file}... ', end='')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    print('done')
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

if __name__ == "__main__":
    pass