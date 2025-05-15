import os, torch
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool


ATLAS_FACTORY = ['AAL_116', 'Aicha_384', 'Gordon_333', 'Brainnetome_264', 'Shaefer_100', 'Shaefer_200', 'Shaefer_400', 'D_160']
BOLD_FORMAT = ['.csv', '.csv', '.tsv', '.csv', '.tsv', '.tsv', '.tsv', '.txt']
DATAROOT = {
    'adni': '/ram/USERS/ziquanw/detour_hcp/data',
    # 'oasis': '/ram/USERS/ziquanw/detour_hcp/data',
    'hcpa': '/ram/USERS/bendan/ACMLab_DATA',
    'ukb': '/ram/USERS/ziquanw/data',
    'hcpya': '/ram/USERS/bendan/ACMLab_DATA',
    'ppmi': '/ram/USERS/jiaqi/benchmark_fmri/data/PPMI',
    'abide': '/ram/USERS/jiaqi/benchmark_fmri/data/ABIDE',
    'neurocon': '/ram/USERS/bendan/ACMLab_DATA/All_Dataset/neurocon/neurocon',
    'taowu': '/ram/USERS/bendan/ACMLab_DATA/All_Dataset/taowu/taowu',
    'sz-diana': '/ram/USERS/ziquanw/data/SZ_data_Schezophrannia_Diana_Jefferies',
}
DATANAME = {
    'adni': 'ADNI_BOLD_SC',
    # 'oasis': 'OASIS_BOLD_SC',
    'hcpa': 'HCP-A-SC_FC',
    'ukb': 'UKB-SC-FC',
    'hcpya': 'HCP-YA-SC_FC',
}
LABEL_NAME_P = {
    'adni': -1, 'oasis': -1, 
    'hcpa': 1, 'hcpya': 1, 
    'ukb': 1,
}

LABEL_REMAP = {
    'adni': {'CN': 'CN', 'SMC': 'CN', 'EMCI': 'CN', 'LMCI': 'AD', 'AD': 'AD'},
    # 'oasis': {'CN': 'CN', 'AD': 'AD'},
}
DISEASE_DATA = ['adni', 'ppmi', 'abide', 'neurocon', 'taowu', 'sz-diana']


def save_data(args):
    fn, fc_root, dname, atlas_name, data_dir, datai = args
    subn = fn.split('_')[0]
    fc = np.array(pd.read_csv(f'{fc_root}/{fn}', delimiter='\t' if fn.endswith('.tsv') else ',').iloc[:,1:])
    label = Path(fn).stem.split('_')[1].lower()
    if 'task' not in label:
        label = Path(fn).stem.split('_')[2].lower()
    np.savez(f'{data_dir}/FC{datai:06d}_{dname}_{subn}_{label}_{atlas_name}', fc)

def main(dnames = ['hcpya', 'ukb', 'hcpa'], atlas_names = ['AAL_116', 'Gordon_333', 'Shaefer_400']):
    data_dir = f'data/{"-".join(dnames)}_{"-".join(atlas_names)}/BOLD'
    os.makedirs(data_dir, exist_ok=True)
    label_name = []
    datai = 0
    for dname in dnames:
        for atlas_name in atlas_names:
            data_root = DATAROOT[dname]
            data_name = DATANAME[dname]
            # subn_p = 0
            # subtask_p = LABEL_NAME_P[dname]
            assert atlas_name in ATLAS_FACTORY, atlas_name
            fc_root = f'{data_root}/{data_name}/{atlas_name}/BOLD'
            # fc_common_rname = None
            # compute FC in getitem
            
            inputs = []
            # for fn in tqdm(os.listdir(fc_root), desc=f'Preload FC {dname} {atlas_name}'):
            for fn in os.listdir(fc_root):
                label = Path(fn).stem.split('_')[1].lower()
                if 'task' not in label:
                    label = Path(fn).stem.split('_')[2].lower()
                if label not in label_name: label_name.append(label)    
                inputs.append([fn, fc_root, dname, atlas_name, data_dir, datai])
                datai += 1

            with Pool(processes=30) as saver_pool:
                list(saver_pool.imap(save_data, tqdm([inp for inp in inputs], desc=f'Prepare FC {dname} {atlas_name}')))

    print(label_name)
    # np.save(f'{data_dir}/label_names.npy', label_name)

if __name__ == '__main__': 
    main()
    # fns = os.listdir('data/hcpya-ukb-hcpa_AAL_116-Gordon_333-Shaefer_400/FC')
    # subjects = ['_'.join(fn.split('_')[1:3]) for fn in tqdm(fns)]
    # subjects = np.unique(subjects)
    # print(len(subjects), subjects[:10])
    # np.save('data/hcpya-ukb-hcpa_AAL_116-Gordon_333-Shaefer_400/subject_names.npy', subjects)