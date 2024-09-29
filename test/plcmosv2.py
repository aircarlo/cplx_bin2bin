"""
Test function to calculate PLCMOSv2 on a set of files
"""
import os, sys
from tqdm import tqdm
import argparse
import yaml
import pandas as pd
from speechmos import plcmos
from utils.utils import file_parse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', required = True, help = 'Experiment ID')
    parser.add_argument('-out_csv', action='store_true', help='Wether to save results into csv file, default False')
    cli_args = parser.parse_args()
    with open(r'config.yaml') as file:
        par = yaml.load(file, Loader=yaml.FullLoader)

    plcmos_dict = []
    clean_plcmos_values = []
    lossy_plcmos_values = []
    inpainted_plcmos_values = []

    clean_dir = os.path.join(par["LOG_PATH"], cli_args.eid, 'CLEAN_PLC2022')
    lossy_dir = os.path.join(par["LOG_PATH"], cli_args.eid, 'LOSSY_PLC2022')
    inpainted_dir = os.path.join(par["LOG_PATH"], cli_args.eid, 'INPAINTED_PLC2022')
    # parse files
    clean_file_list = file_parse(clean_dir, 'wav', return_fullpath=False)
    lossy_file_list = file_parse(lossy_dir, 'wav', return_fullpath=False)
    inpainted_file_list = file_parse(inpainted_dir, 'wav', return_fullpath=False)

    # check consistency
    assert set(clean_file_list) == set(lossy_file_list), f"error: clean_file_list differs from lossy_file_list"
    assert set(clean_file_list) == set(inpainted_file_list), f"error: clean_file_list differs from inpainted_file_list"
    assert set(lossy_file_list) == set(inpainted_file_list), f"error: lossy_file_list differs from inpainted_file_list"

    print('Compute PLCMOSv2 score...')

    for f in tqdm(clean_file_list):
        
        clean_p = plcmos.run(os.path.join(clean_dir, f), sr=16000)
        lossy_p = plcmos.run(os.path.join(lossy_dir, f), sr=16000)
        inpainted_p = plcmos.run(os.path.join(inpainted_dir, f), sr=16000)

        clean_plcmos_values.append(clean_p['plcmos'])
        lossy_plcmos_values.append(lossy_p['plcmos'])
        inpainted_plcmos_values.append(inpainted_p['plcmos'])

        plcmos_dict.append({'filename': f,
                            'clean_plcmos': clean_p['plcmos'],
                            'lossy_plcmos': lossy_p['plcmos'],
                            'inpainted_plcmos': inpainted_p['plcmos']})
    
    print('Mean PLCMOS value')
    print(f'clean: {sum(clean_plcmos_values)/len(clean_plcmos_values)}')
    print(f'lossy: {sum(lossy_plcmos_values)/len(lossy_plcmos_values)}')
    print(f'inpainted: {sum(inpainted_plcmos_values)/len(inpainted_plcmos_values)}')

    if cli_args.out_csv:
        df = pd.DataFrame.from_dict(plcmos_dict)
        csv_path = os.path.join(par["LOG_PATH"], cli_args.eid, 'PLCMOSv2.csv')
        df.to_csv (csv_path, index=False, header=True)
        print(f'{csv_path} saved')