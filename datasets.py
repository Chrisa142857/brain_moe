import os, torch, difflib
from scipy.io import loadmat
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
from sklearn.model_selection import KFold
# from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
# import networkx as nx
from tqdm import tqdm, trange
from prepare_pretrain_data import main as prepare_pretrain_data

ATLAS_ROI_N = {
    'AAL_116': 116,
    'Gordon_333': 333,
    'Shaefer_100': 100,
    'Shaefer_200': 200,
    'Shaefer_400': 400,
    'D_160': 160
}
ATLAS_FACTORY = ['AAL_116', 'Aicha_384', 'Gordon_333', 'Brainnetome_264', 'Shaefer_100', 'Shaefer_200', 'Shaefer_400', 'D_160']
BOLD_FORMAT = ['.csv', '.csv', '.tsv', '.csv', '.tsv', '.tsv', '.tsv', '.txt']
DATAROOT = {
    'adni': '../detour_hcp/data',
    # 'oasis': '../detour_hcp/data',
    'hcpa': '../Lab_DATA',
    'ukb': '../data',
    'hcpya': '../Lab_DATA',
    'ppmi': '../benchmark_fmri/data/PPMI',
    'abide': '../benchmark_fmri/data/ABIDE',
    'neurocon': '../Lab_DATA/All_Dataset/neurocon/neurocon',
    'taowu': '../Lab_DATA/All_Dataset/taowu/taowu',
    'sz-diana': '../data/SZ_data_Schezophrannia_Diana_Jefferies',
    'fmrieeg': '../brain_moe/data/fmrieeg-Shaefer_400',
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
class PretrainDataset(Dataset): 
    def __init__(self, atlas_names=['AAL_116', 'Gordon_333', 'Shaefer_400'],
                 dnames=['hcpya', 'ukb', 'hcpa'],
                 tgt_atlas_name='AAL_116', # For the ease of data batching, one atlas each dataLoader
                ) -> None:
        data_dir = f'data/{"-".join(dnames)}_{"-".join(atlas_names)}'
        self.data_dir = data_dir
        self.data_fns = [fn for fn in os.listdir(f'{data_dir}/FC') if tgt_atlas_name in fn]
        self.label_names = list(np.load(f'{data_dir}/label_names.npy'))
        self.subject = ['_'.join(fn.split('_')[1:3]) for fn in self.data_fns]
        self.data_subj = np.unique(self.subject)
        self.atlas_names = atlas_names
        self.labels = np.array([self.label_names.index(fn.split('_')[3]) for fn in self.data_fns])
        self.label_onehot = np.zeros([len(self.labels), len(self.label_names)]).astype(np.float32)
        self.label_onehot[np.arange(len(self.labels)), self.labels] = 1
        # expert_tags = ['_'.join(fn.split('_')[-3:]).split('.')[0] for fn in self.data_fns]
        print("Data num", len(self), "FC shape (N x N)", np.load(f'{self.data_dir}/FC/{self.data_fns[0]}')['arr_0'].shape, "Label name", self.label_names)
        self.expert_tags = [f'{self.label_names[li]}_{tgt_atlas_name}' for li in range(len(self.label_names))]
        self.cached_data = [None for _ in range(len(self))]

    def __getitem__(self, index):
        if self.cached_data[index] is None:
            fn = f'{self.data_dir}/FC/{self.data_fns[index]}'
            atlas_name = '_'.join(self.data_fns[index].split('_')[-2:]).split('.')[0]
            x = np.load(fn)['arr_0']
            subjn = self.subject[index]
            y = self.label_onehot[index]
            x[np.isnan(x)] = 0
            x[np.isinf(x)] = 0
            data = {
                'x': x.astype(np.float32),
                'y': y,
                'subject': subjn,
                'expert_tag': f'{self.label_names[self.labels[index]]}_{atlas_name}',
            }
            self.cached_data[index] = data
        return self.cached_data[index]

    def __len__(self):
        return len(self.data_fns)


def dataloader_generator(batch_size=4, num_workers=8, train_ratio=0.8, **kargs):
    dataset = PretrainDataset(**kargs)
    all_subjects = dataset.data_subj
    np.random.shuffle(all_subjects)
    train_subj_n = int(len(all_subjects)*train_ratio)
    train_subjects = all_subjects[:train_subj_n]
    val_subjects = all_subjects[train_subj_n:]
    # Filter dataset based on training and validation subjects
    train_data = [di for di, subj in enumerate(dataset.subject) if subj in train_subjects]
    val_data = [di for di, subj in enumerate(dataset.subject) if subj in val_subjects]
    print(f'Train {len(train_subjects)} subjects, Val {len(val_subjects)} subjects, len(train_data)={len(train_data)}, len(val_data)={len(val_data)}')
    train_dataset = Subset(dataset, train_data)
    valid_dataset = Subset(dataset, val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, dataset

class NeuroNetworkDataset(Dataset):

    def __init__(self, atlas_name='AAL_116',
                 dname='hcpa',
                # node_attr = 'SC', adj_type = 'FC',
                transform = [],
                transform_tag = [],
                # fc_winsize = 500,
                # fc_winoverlap = 0,
                # fc_th = 0.5,
                # sc_th = 0.1
                ) -> None:
        # default_fc_th = 0.5
        # default_sc_th = 0.1
        data_root = DATAROOT[dname]
        data_name = DATANAME[dname]
        self.transform = transform
        self.transform_tag = transform_tag
        self.data_root = f"{data_root}/{data_name}"
        # self.fc_winsize = fc_winsize
        # self.fc_th = fc_th
        # self.sc_th = sc_th
        self.dname = dname
        subn_p = 0
        subtask_p = LABEL_NAME_P[dname]
        # subdir_p = 2
        # bold_format = BOLD_FORMAT[ATLAS_FACTORY.index(atlas_name)]
        # fc_format = '.csv'
        assert atlas_name in ATLAS_FACTORY, atlas_name
        bold_root = f'{self.data_root}/{atlas_name}/BOLD'
        fc_root = f'{self.data_root}/{atlas_name}/FC'
        # sc_root = f'{self.data_root}/ALL_SC'
        atlas_name = CORRECT_ATLAS_NAME(atlas_name)
        data_dir = f'{dname}-{atlas_name}'
        # if self.fc_th == default_fc_th and self.sc_th == default_sc_th:
        #     data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}'
        # else:
        #     data_dir = f"{dname}-{atlas_name}-BOLDwin{fc_winsize}-FCth{str(self.fc_th).replace('.', '')}SCth{str(self.sc_th).replace('.', '')}"
        os.makedirs(f'data/{data_dir}', exist_ok=True)
        if not os.path.exists(f'data/{data_dir}/raw.pt'):
            fc_subs = [fn.split('_')[subn_p] for fn in os.listdir(fc_root)]
            subs = np.unique(fc_subs)
            # sc_subs = [fn.split('_')[subn_p] for fn in os.listdir(sc_root)]
            # subs = np.intersect1d(fc_subs, sc_subs)
            self.all_sc = {}
            self.all_fc = {}
            self.label_name = []
            self.sc_common_rname = None
            # for fn in tqdm(os.listdir(sc_root), desc='Load SC'):
            #     subn = fn.split('_')[subn_p]
            #     if subn in subs:
            #         sc, rnames, _ = load_sc(f"{sc_root}/{fn}", atlas_name)
            #         if self.sc_common_rname is None: self.sc_common_rname = rnames
            #         if self.sc_common_rname is not None: 
            #             _, rid, _ = np.intersect1d(rnames, self.sc_common_rname, return_indices=True)
            #             self.all_sc[subn] = sc[rid, :][:, rid]
            #         else:
            #             self.all_sc[subn] = sc
            self.fc_common_rname = None
            # compute FC in getitem
            self.data = {'bold': [], 'subject': [], 'label': [], 'winid': []}
            for fn in tqdm(os.listdir(bold_root), desc='Load BOLD'):
                if fn.split('_')[subn_p] in subs:
                    bolds = load_bold(f"{bold_root}/{fn}")
                    subn = fn.split('_')[subn_p]
                    # if self.fc_common_rname is None: self.fc_common_rname = rnames
                    # if self.fc_common_rname is not None: 
                    #     _, rid, _ = np.intersect1d(rnames, self.fc_common_rname, return_indices=True)
                    #     bolds = [b[rid] for b in bolds]
                
                    label = Path(fn).stem.split('_')[subtask_p]
                    if dname in ['adni', 'oasis']:
                        if label not in LABEL_REMAP[dname]: continue
                        label = LABEL_REMAP[dname][label]
                    if label not in self.label_name: self.label_name.append(label)
                    self.data['bold'].extend(bolds) # N x T
                    self.data['subject'].extend([subn for _ in bolds])
                    self.data['label'].extend([self.label_name.index(label) for _ in bolds])
                    self.data['winid'].extend([i for i in range(len(bolds))])

            # if self.sc_common_rname is not None and self.fc_common_rname is not None:
            #     self.sc_common_rname = [rn.strip() for rn in self.sc_common_rname]
            #     self.fc_common_rname = [rn.strip() for rn in self.fc_common_rname]
            #     common_rname, sc_rid, fc_rid = np.intersect1d(self.sc_common_rname, self.fc_common_rname, return_indices=True)
            #     for sub in self.all_sc:
            #         self.all_sc[sub] = self.all_sc[sub][:, sc_rid][sc_rid, :]
            #     for i in range(len(self.data['subject'])):
            #         self.data['bold'][i] = self.data['bold'][i][fc_rid]
            #     self.sc_common_rname = common_rname
            #     self.fc_common_rname = common_rname
            # self.data['all_sc'] = self.all_sc
            self.data['label_name'] = self.label_name
            torch.save(self.data, f'data/{data_dir}/raw.pt')
        
        self.data = torch.load(f'data/{data_dir}/raw.pt')
        # self.all_sc = self.data['all_sc']
        # self.adj_type = adj_type
        # self.node_attr = node_attr
        self.atlas_name = atlas_name
        self.subject = np.array(self.data['subject'])
        # self.data['label'] = np.array(self.data['label'])
        self.label_names = list(self.data['label_name'])
        self.data_subj = np.unique(self.subject)
        self.node_num = len(self.data['bold'][0])
        self.cached_data = [None for _ in range(len(self.subject))]
        self.label_remap = None
        if 'task-rest' in self.data['label_name'] or 'task-REST' in self.data['label_name']:
            restli = [i for i, l in enumerate(self.data['label_name']) if 'rest' in l.lower()]
            assert len(restli) == 1, self.data['label_name']
            restli = restli[0]
            nln = list(self.data['label_name'])
            nln[0] = self.data['label_name'][restli]
            nln[restli] = self.data['label_name'][0]
            self.data['label_name'] = nln
            self.label_remap = {restli: 0, 0: restli}

        if os.path.exists(f'../data/meta_data/{dname.upper()}_metadata.csv'):
            meta_data = pd.read_csv(f'../data/meta_data/{dname.upper()}_metadata.csv')
            self.subj2sex = {
                subj: np.unique(meta_data[meta_data['Subject']==subj]['Sex']).item()
            for subj in self.data_subj if subj in list(meta_data['Subject'])}
            self.sex_label = {'M': 0, 'F': 1}
            self.subj2sex = {k: self.sex_label[v] for k, v in self.subj2sex.items()}
            self.subj2age = {
                subj: str(np.unique(meta_data[meta_data['Subject']==subj]['Age']).item())
            for subj in self.data_subj if subj in list(meta_data['Subject'])}
            self.subj2age = {
                k: float(v) if '-' not in v else np.array(v.split('-')).astype(np.int32).mean()
            for k, v in self.subj2age.items()}
            self.subj2age = {
                k: torch.tensor(v).float().reshape(1, 1)
            for k, v in self.subj2age.items()}
            if len(self.subj2age) > 0:
                # self.age_max = np.max(list(self.subj2age.values()))
                # self.age_min = np.min(list(self.subj2age.values()))
                # self.subj2age = {k: (v-self.age_min)/(self.age_max-self.age_min) for k, v in self.subj2age.items()}
                self.subj2age = {k: v if v <= 150 else v/12 for k, v in self.subj2age.items()}
            sex_label = np.array(list(self.subj2sex.values()))
            print(f"Subject sex dist: [{sum(sex_label)} + {len(sex_label) - sum(sex_label)} = {len(sex_label)}]")
        else:
            self.subj2sex = {}
            self.subj2age = {}
        
        print("Data num", len(self), "BOLD shape (N x T)", self.data['bold'][0].shape, "Label name", self.data['label_name'])
        # if self.transform is not None:
        #     processed_fn = f'processed_adj{self.adj_type}x{self.node_attr}_FCth{self.fc_th}SCth{self.sc_th}_{type(self.transform).__name__}{self.transform.k}'.replace('.', '')
        #     if not os.path.exists(f'data/{data_dir}/{processed_fn}.pt'):
        #         for _ in tqdm(self, desc='Processing'):
        #             pass
                
        #         torch.save(self.cached_data, f'data/{data_dir}/{processed_fn}.pt')
        #     self.cached_data = torch.load(f'data/{data_dir}/{processed_fn}.pt')
        
        # for _ in tqdm(self, desc='Preloading'):
        #     pass
        
        self.dname_list = None
        self.nclass_list = [len(self.label_names)]
        self.expert_embeds = [0 for _ in range(len(self))]
        
    def __getitem__(self, index):
        if self.cached_data[index] is None:
            subjn = self.subject[index]
            x = torch.corrcoef(self.data['bold'][index])
            # sc = self.all_sc[subjn]
            # edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
            # edge_index_sc = torch.stack(torch.where(sc > self.sc_th))
            # if self.adj_type == 'FC':
            #     edge_index = edge_index_fc
            #     # adj = torch.sparse_coo_tensor(indices=edge_index_fc, values=fc[edge_index_fc[0], edge_index_fc[1]], size=(self.node_num, self.node_num))
            # else:
            #     edge_index = edge_index_sc
            #     # adj = torch.sparse_coo_tensor(indices=edge_index_sc, values=sc[edge_index_sc[0], edge_index_sc[1]], size=(self.node_num, self.node_num))
            # if self.node_attr=='FC':
            #     x = fc
            # elif self.node_attr=='BOLD':
            #     x = self.data['bold'][index]
            # elif self.node_attr=='SC':
            #     x = sc
            # elif self.node_attr=='ID':
            #     x = torch.arange(self.node_num).float()[:, None]
        
            x[x.isnan()] = 0
            x[x.isinf()] = 0
            data = {
                # 'edge_index': edge_index,
                'x': x,
                'y': self.data['label'][index],
                'sex': self.subj2sex[subjn] if subjn in self.subj2sex else -1,
                'age': self.subj2age[subjn] if subjn in self.subj2age else torch.tensor([[-1]]).float(),
                # 'edge_index_fc': edge_index_fc,
                # 'edge_index_sc': edge_index_sc
            }
            # if self.transform is not None:
            #     new_data = self.transform(data)
            #     # self.cached_data[index] = new_data
            #     for key in new_data:
            #         data[key] = new_data[key]
                    
            # adj_fc = torch.zeros(x.shape[0], x.shape[0]).bool()
            # adj_fc[edge_index_fc[0], edge_index_fc[1]] = True
            # adj_sc = torch.zeros(x.shape[0], x.shape[0]).bool()
            # adj_sc[edge_index_sc[0], edge_index_sc[1]] = True
            # adj_fc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            # adj_sc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            # data['adj_fc'] = adj_fc[None]
            # data['adj_sc'] = adj_sc[None]
            
            self.cached_data[index] = data
        data = self.cached_data[index]
        for t, tn in zip(self.transform, self.transform_tag):
            data[tn] = t(data)
        if self.expert_embeds[index] != 0:
            data['expert_embed'] = self.expert_embeds[index]
        if self.label_remap is not None:
            if data['y'] in self.label_remap:
                data['y'] = self.label_remap[data['y']]
        return data

    def __len__(self):
        return len(self.cached_data)


def load_fc(fpath):
    mat = pd.read_csv(fpath)
    mat = torch.from_numpy(mat[:, 1:].astype(np.float32))
    rnames = mat[:, 0]
    return mat, rnames, fpath.split('/')[-1]

def load_sc(path, atlas_name):
    if not path.endswith('.mat') and not path.endswith('.txt'):
        matfns = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mat')]
        txtfns = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
        return load_sc(matfns[0] if len(matfns) > 0 else txtfns[0], atlas_name)
    if path.endswith('.mat'):
        fpath = f"{path}"
        sc_mat = loadmat(fpath)
        mat = sc_mat[f"{atlas_name.lower().replace('_','')}_sift_radius2_count_connectivity"]
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = sc_mat[f"{atlas_name.lower().replace('_','')}_region_labels"]
    elif path.endswith('.txt'):
        fpath = f"{path}"
        mat = np.loadtxt(fpath)
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = None
    return mat, rnames, path.split('/')[-1]

def load_bold(path):
    if not path.endswith('.txt'):
        bold_pd = pd.read_csv(path) if not path.endswith('.tsv') else pd.read_csv(path, sep='\t')
        if isinstance(np.array(bold_pd)[0, 1], str):
            rnames = list(bold_pd.columns[1:])
            bold = torch.from_numpy(np.array(bold_pd)[:, 1:]).float().T
        else:
            rnames = list(bold_pd.columns)
            bold = torch.from_numpy(np.array(bold_pd)).float().T
            if bold.shape[0] > ATLAS_ROI_N[path.split('/')[-3]]:
                bold = bold[1:]
            elif bold.shape[0] < ATLAS_ROI_N[path.split('/')[-3]]:
                print("Error bold.shape[0] < ATLAS_ROI_N[path.split('/')[-3]]", bold.shape[0], ATLAS_ROI_N[path.split('/')[-3]])
                exit()
    else:
        rnames = None
        bold = torch.from_numpy(np.loadtxt(path)).float().T
    return [bold]

def CORRECT_ATLAS_NAME(n):
    if n == 'Brainnetome_264': return 'Brainnetome_246'
    if 'Shaefer_' in n: return n.replace('Shaefer', 'Schaefer')
    return n

def Schaefer_SCname_match_FCname(scns, fcns):
    '''
    TODO: Align Schaefer atlas region name of SC and FC
    '''
    match = []
    def get_overlap(s1, s2):
        s = difflib.SequenceMatcher(None, s1, s2)
        pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
        return s1[pos_a:pos_a+size]
    
    for fcn in fcns:
        fcn = fcn.replace('17Networks_', '')
        fcn_split = fcn.split('_')
        sc_overlap_len = []
        for scn in scns:
            scn_split = scn.split('_')
            if scn_split[0] != fcn_split[0] or scn_split[-1] != fcn_split[-1]:
                continue
            sc_overlap_len.append(sum([len(get_overlap(scn_split[i], fcn_split[i])) for i in range(1, len(scn_split)-1)]))
        match.append()

    return match

def ttest_fc(fcs1, fcs2, thr=0.05):
    from scipy import stats
    print(fcs1.shape, fcs2.shape)
    significant_fc = []
    ps = []
    for i in trange(fcs1.shape[1]):
        for j in range(i+1, fcs1.shape[2]):
            a = fcs1[:, i, j].numpy()
            b = fcs2[:, i, j].numpy()
            p = stats.ttest_ind(a, b).pvalue
            if p < thr: 
                significant_fc.append([i, j])
                ps.append(p)
    significant_fc = torch.LongTensor(significant_fc)
    ps = torch.FloatTensor(ps)
    print(significant_fc.shape)
    return significant_fc, ps

import re
class Dataset_PPMI_ABIDE(Dataset):
    def __init__(self, atlas_name='AAL_116', # multi-atlas not available 
                 dname='ppmi',
                transform = [],
                transform_tag = [],
                # node_attr = 'FC', adj_type = 'FC',
                # transform = None,
                # fc_winsize = 137, # not implement
                # fc_winoverlap = 0, # not implement
                # fc_th = 0.5,
                # sc_th = 0.1, **kargs
                ):
        super(Dataset_PPMI_ABIDE, self).__init__()
        # self.adj_type = adj_type
        self.transform = transform
        self.transform_tag = transform_tag
        # self.node_attr = node_attr
        self.atlas_name = atlas_name
        # self.fc_th = fc_th
        # self.sc_th = sc_th
        # self.fc_winsize = fc_winsize
        self.node_num = 116
        self.label_remap = None
        self.root_dir = DATAROOT[dname]
        self.dname = dname
        self.data = []
        self.labels = []
        self.data_path = []
        self.subject = []
        self.label_names = [None for _ in range(4)]
        
        default_fc_th = 0.5
        default_sc_th = 0.1
        data_dir = f'{dname}-{atlas_name}'
        # if self.fc_th == default_fc_th and self.sc_th == default_sc_th:
        #     data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}'
        # else:
        #     data_dir = f"{dname}-{atlas_name}-BOLDwin{fc_winsize}-FCth{str(self.fc_th).replace('.', '')}SCth{str(self.sc_th).replace('.', '')}"
        os.makedirs(f'data/{data_dir}', exist_ok=True)
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'AAL116_features_timeseries' in file:
                    file_path = os.path.join(subdir, file)
                    self.data_path.append(file_path)
                    label, label_name = self.get_label(subdir)
                    self.labels.append(label)
                    self.label_names[label] = label_name
                    self.subject.append(subdir.split('/')[-1])
        self.label_names = [l for l in self.label_names if l is not None]
        self.cached_data = [None for _ in range(len(self.data_path))]
        self.data_subj = np.unique(self.subject)
        if os.path.exists(f'../data/meta_data/{dname.upper()}_metadata.csv'):
            meta_data = pd.read_csv(f'../data/meta_data/{dname.upper()}_metadata.csv')
            self.subj2sex = {
                subj: np.unique(meta_data[meta_data['Subject']==int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0])]['Sex']).item()
            for subj in self.data_subj if int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0]) in list(meta_data['Subject'])}
            self.sex_label = {'M': 0, 'F': 1}
            self.subj2sex = {k: self.sex_label[v] for k, v in self.subj2sex.items()}
            self.subj2age = {
                subj: torch.tensor(np.unique(meta_data[meta_data['Subject']==int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0])]['Age']).item()).float().reshape(1, 1)
            for subj in self.data_subj if int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0]) in list(meta_data['Subject'])}
            # if len(self.subj2age) > 0:
            #     self.age_max = np.max(list(self.subj2age.values()))
            #     self.age_min = np.min(list(self.subj2age.values()))
            #     self.subj2age = {k: (v-self.age_min)/(self.age_max-self.age_min) for k, v in self.subj2age.items()}
        else:
            self.subj2sex = {}
            self.subj2age = {}
        # if self.transform is not None:
        #     processed_fn = f'processed_adj{self.adj_type}x{self.node_attr}_FCth{self.fc_th}SCth{self.sc_th}_{type(self.transform).__name__}{self.transform.k}'.replace('.', '')
        #     if not os.path.exists(f'data/{data_dir}/{processed_fn}.pt'):
        #         for _ in tqdm(self, desc='Processing'):
        #             pass
                
        #         torch.save(self.cached_data, f'data/{data_dir}/{processed_fn}.pt')
        #     self.cached_data = torch.load(f'data/{data_dir}/{processed_fn}.pt')
        
        # for _ in tqdm(self, desc='Preload data'):
        #     pass
        self.nclass_list = [len(self.label_names)]
        self.expert_embeds = [0 for _ in range(len(self))]
        print("Data num", len(self), "Label name", self.label_names)
            
    def get_label(self, subdir):
        if 'control' in subdir:
            return 0, 'CN'
        elif 'patient' in subdir:
            return 1, 'Autism' if 'ABIDE' in subdir else 'PD'
        elif 'prodromal' in subdir:
            return 2, 'ppmi-2'
        elif 'swedd' in subdir:
            return 3, 'ppmi-3'
        else:
            assert False, subdir
        
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, index):
        # label = self.labels[index]
        # data = (features - torch.mean(features, axis=0, keepdims=True)) / torch.std(features, axis=0, keepdims=True)
        if self.cached_data[index] is None:
            features = loadmat(self.data_path[index])['data'].T
            features = torch.from_numpy(features).float()
            x = torch.nan_to_num(features)
            x = torch.corrcoef(x)
            subjn = self.subject[index]
            # edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
            # edge_index_sc = None
            # edge_index_sc = edge_index_fc
            # if self.adj_type == 'FC':
            #     edge_index = edge_index_fc
            #     # adj = torch.sparse_coo_tensor(indices=edge_index_fc, values=fc[edge_index_fc[0], edge_index_fc[1]], size=(self.node_num, self.node_num))
            # else:
            #     assert False, "Not implement"
                # adj = torch.sparse_coo_tensor(indices=edge_index_sc, values=sc[edge_index_sc[0], edge_index_sc[1]], size=(self.node_num, self.node_num))
            # if self.node_attr=='FC':
            #     x = fc
            # elif self.node_attr=='BOLD':
            #     x = x[:, :self.fc_winsize]
            #     if x.shape[1] < self.fc_winsize: 
            #         x = torch.cat([x, torch.zeros(x.shape[0], self.fc_winsize-x.shape[1])], 1)
                    
            # elif self.node_attr=='SC':
            #     assert False, "Not implement"
            # elif self.node_attr=='ID':
            #     x = torch.arange(self.node_num).float()[:, None]
        
            x[x.isnan()] = 0
            x[x.isinf()] = 0
            data = {
                # 'edge_index': edge_index,
                'x': x,
                'y': self.labels[index],
                'sex': self.subj2sex[subjn] if subjn in self.subj2sex else -1,
                'age': self.subj2age[subjn] if subjn in self.subj2age else torch.tensor([[-1]]).float(),
                # 'edge_index_fc': edge_index_fc,
                # 'edge_index_sc': edge_index_sc
            }
            # if self.transform is not None:
            #     new_data = self.transform(data)
            #     # self.cached_data[index] = new_data
            #     for key in new_data:
            #         data[key] = new_data[key]
                    
            # adj_fc = torch.zeros(x.shape[0], x.shape[0]).bool()
            # adj_fc[edge_index_fc[0], edge_index_fc[1]] = True
            # adj_sc = torch.zeros(x.shape[0], x.shape[0]).bool()
            # adj_sc[edge_index_sc[0], edge_index_sc[1]] = True
            # adj_fc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            # adj_sc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            # data['adj_fc'] = adj_fc[None]
            # data['adj_sc'] = adj_sc[None]
            
            self.cached_data[index] = data
        data = self.cached_data[index]
        for t, tn in zip(self.transform, self.transform_tag):
            data[tn] = t(data)
        if self.expert_embeds[index] != 0:
            data['expert_embed'] = self.expert_embeds[index]
        if self.label_remap is not None:
            if data['y'] in self.label_remap:
                data['y'] = self.label_remap[data['y']]
        return data



class Dataset_SZ(Dataset):
    def __init__(self, atlas_name='AAL_116', # multi-atlas not available 
                 dname='sz-diana',
                transform = [],
                transform_tag = [],
                # node_attr = 'FC', adj_type = 'FC',
                # transform = None,
                # fc_winsize = 137, # not implement
                # fc_winoverlap = 0, # not implement
                # fc_th = 0.5,
                # sc_th = 0.1, **kargs
                ):
        super(Dataset_SZ, self).__init__()
        # self.adj_type = adj_type
        self.transform = transform
        self.transform_tag = transform_tag
        # self.transform = transform
        # self.node_attr = node_attr
        self.atlas_name = atlas_name
        # self.fc_th = fc_th
        # self.sc_th = sc_th
        # self.fc_winsize = fc_winsize
        self.node_num = 116
        self.label_remap = None
        self.root_dir = DATAROOT[dname]
        self.dname = dname
        self.data = []
        self.labels = []
        self.data_path = []
        self.subject = []
        self.label_names = [None for _ in range(4)]
        
        # default_fc_th = 0.5
        # default_sc_th = 0.1
        data_dir = f'{dname}-{atlas_name}'
        # if self.fc_th == default_fc_th and self.sc_th == default_sc_th:
        #     data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}'
        # else:
        #     data_dir = f"{dname}-{atlas_name}-BOLDwin{fc_winsize}-FCth{str(self.fc_th).replace('.', '')}SCth{str(self.sc_th).replace('.', '')}"
        os.makedirs(f'data/{data_dir}', exist_ok=True)
        for subdir, _, files in os.walk(self.root_dir):
            if 'Nonconverters2' in subdir: continue
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(subdir, file)
                    self.data_path.append(file_path)
                    label, label_name = self.get_label(subdir)
                    self.labels.append(label)
                    self.label_names[label] = label_name
                    self.subject.append(file)
        self.label_names = [l for l in self.label_names if l is not None]
        self.cached_data = [None for _ in range(len(self.data_path))]
        self.data_subj = np.unique(self.subject)
        if os.path.exists(f'../data/meta_data/{dname.upper()}_metadata.csv'):
            meta_data = pd.read_csv(f'../data/meta_data/{dname.upper()}_metadata.csv')
            self.subj2sex = {
                subj: np.unique(meta_data[meta_data['Subject']==int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0])]['Sex']).item()
            for subj in self.data_subj if int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0]) in list(meta_data['Subject'])}
            self.sex_label = {'M': 0, 'F': 1}
            self.subj2sex = {k: self.sex_label[v] for k, v in self.subj2sex.items()}
            self.subj2age = {
                subj: torch.tensor(np.unique(meta_data[meta_data['Subject']==int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0])]['Age']).item()).float().reshape(1, 1)
            for subj in self.data_subj if int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0]) in list(meta_data['Subject'])}
            # if len(self.subj2age) > 0:
            #     self.age_max = np.max(list(self.subj2age.values()))
            #     self.age_min = np.min(list(self.subj2age.values()))
            #     self.subj2age = {k: (v-self.age_min)/(self.age_max-self.age_min) for k, v in self.subj2age.items()}
        else:
            self.subj2sex = {}
            self.subj2age = {}
        # if self.transform is not None:
        #     processed_fn = f'processed_adj{self.adj_type}x{self.node_attr}_FCth{self.fc_th}SCth{self.sc_th}_{type(self.transform).__name__}{self.transform.k}'.replace('.', '')
        #     if not os.path.exists(f'data/{data_dir}/{processed_fn}.pt'):
        #         for _ in tqdm(self, desc='Processing'):
        #             pass
                
        #         torch.save(self.cached_data, f'data/{data_dir}/{processed_fn}.pt')
        #     self.cached_data = torch.load(f'data/{data_dir}/{processed_fn}.pt')
        
        # for _ in tqdm(self, desc='Preload data'):
        #     pass
        self.nclass_list = [len(self.label_names)]
        self.expert_embeds = [0 for _ in range(len(self))]
        print("Data num", len(self), "Label name", self.label_names)
            
    def get_label(self, subdir):
        if 'Nonconverters2' in subdir:
            return 0, 'CN'
        elif 'Unaffected2' in subdir:
            return 0, 'CN' # -high-risk
        elif 'Converters2' in subdir:
            return 1, 'SZ'
        else:
            assert False, subdir
        
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, index):
        # label = self.labels[index]
        # data = (features - torch.mean(features, axis=0, keepdims=True)) / torch.std(features, axis=0, keepdims=True)
        if self.cached_data[index] is None:
            features = np.loadtxt(self.data_path[index]).T # N x T
            features = torch.from_numpy(features).float()
            x = torch.nan_to_num(features)
            x = torch.corrcoef(x)
            subjn = self.subject[index]
            # edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
            # edge_index_sc = None
            # edge_index_sc = edge_index_fc
            # if self.adj_type == 'FC':
            #     edge_index = edge_index_fc
            #     # adj = torch.sparse_coo_tensor(indices=edge_index_fc, values=fc[edge_index_fc[0], edge_index_fc[1]], size=(self.node_num, self.node_num))
            # else:
            #     assert False, "Not implement"
            #     # adj = torch.sparse_coo_tensor(indices=edge_index_sc, values=sc[edge_index_sc[0], edge_index_sc[1]], size=(self.node_num, self.node_num))
            # if self.node_attr=='FC':
            #     x = fc
            # elif self.node_attr=='BOLD':
            #     x = x[:, :self.fc_winsize]
            #     if x.shape[1] < self.fc_winsize: 
            #         x = torch.cat([x, torch.zeros(x.shape[0], self.fc_winsize-x.shape[1])], 1)
                    
            # elif self.node_attr=='SC':
            #     assert False, "Not implement"
            # elif self.node_attr=='ID':
            #     x = torch.arange(self.node_num).float()[:, None]
        
            x[x.isnan()] = 0
            x[x.isinf()] = 0
            data = {
                # 'edge_index': edge_index,
                'x': x,
                'y': self.labels[index],
                'sex': self.subj2sex[subjn] if subjn in self.subj2sex else -1,
                'age': self.subj2age[subjn] if subjn in self.subj2age else torch.tensor([[-1]]).float(),
                # 'edge_index_fc': edge_index_fc,
                # 'edge_index_sc': edge_index_sc
            }
            # if self.transform is not None:
            #     new_data = self.transform(data)
            #     # self.cached_data[index] = new_data
            #     for key in new_data:
            #         data[key] = new_data[key]
                    
            # adj_fc = torch.zeros(x.shape[0], x.shape[0]).bool()
            # adj_fc[edge_index_fc[0], edge_index_fc[1]] = True
            # adj_sc = torch.zeros(x.shape[0], x.shape[0]).bool()
            # adj_sc[edge_index_sc[0], edge_index_sc[1]] = True
            # adj_fc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            # adj_sc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            # data['adj_fc'] = adj_fc[None]
            # data['adj_sc'] = adj_sc[None]
            
            self.cached_data[index] = data
        data = self.cached_data[index]
        for t, tn in zip(self.transform, self.transform_tag):
            data[tn] = t(data)
        # data['expert_embed'] = self.expert_embeds[index]
        if self.expert_embeds[index] != 0:
            data['expert_embed'] = self.expert_embeds[index]
        if self.label_remap is not None:
            if data['y'] in self.label_remap:
                data['y'] = self.label_remap[data['y']]
        return data


class Dataset_FMRIEEG(Dataset):
    def __init__(self, atlas_name='Shaefer_400', # multi-atlas not available 
                 dname='fmrieeg',
                transform = [],
                transform_tag = [],
                # node_attr = 'FC', adj_type = 'FC',
                # transform = None,
                # fc_winsize = 137, # not implement
                # fc_winoverlap = 0, # not implement
                # fc_th = 0.5,
                # sc_th = 0.1, **kargs
                ):
        super(Dataset_FMRIEEG, self).__init__()
        # self.adj_type = adj_type
        self.transform = transform
        self.transform_tag = transform_tag
        # self.transform = transform
        # self.node_attr = node_attr
        self.atlas_name = atlas_name
        # self.fc_th = fc_th
        # self.sc_th = sc_th
        # self.fc_winsize = fc_winsize
        self.node_num = 400
        self.label_remap = None
        self.root_dir = DATAROOT[dname]
        self.dname = dname
        self.data = []
        self.labels = []
        self.data_path = []
        self.subject = []
        self.eeg_path = []
        self.label_names = ['task-checker', 'task-dme', 'task-dmh', 'task-inscapes', 'task-monkey', 'task-peer', 'task-rest', 'task-tp']
        
        # data_dir = f'{dname}-{atlas_name}'
        # os.makedirs(f'data/{data_dir}', exist_ok=True)
        for fn in os.listdir(self.root_dir+'/BOLD'):
        
            file_path = os.path.join(self.root_dir+'/BOLD', fn)
            self.data_path.append(file_path)
            label, label_name = self.get_label(fn)
            self.labels.append(label)
            self.subject.append(fn.split('_')[0])
            self.eeg_path.append(file_path.replace('BOLD', 'EEG').replace('bold.tsv', 'eeg.npy'))

        self.cached_data = [None for _ in range(len(self.data_path))]
        self.data_subj = np.unique(self.subject)
        if os.path.exists(f'../data/meta_data/{dname.upper()}_metadata.csv'):
            meta_data = pd.read_csv(f'../data/meta_data/{dname.upper()}_metadata.csv')
            self.subj2sex = {
                subj: np.unique(meta_data[meta_data['Subject']==subj]['Sex']).item()
            for subj in self.data_subj if subj in list(meta_data['Subject'])}
            self.sex_label = {'Male': 0, 'Female': 1}
            self.subj2sex = {k: self.sex_label[v] for k, v in self.subj2sex.items()}
            self.subj2age = {
                subj: torch.tensor(np.unique(meta_data[meta_data['Subject']==subj]['Age']).item()).float().reshape(1)
            for subj in self.data_subj if subj in list(meta_data['Subject'])}
            # if len(self.subj2age) > 0:
            #     self.age_max = np.max(list(self.subj2age.values()))
            #     self.age_min = np.min(list(self.subj2age.values()))
            #     self.subj2age = {k: (v-self.age_min)/(self.age_max-self.age_min) for k, v in self.subj2age.items()}
        else:
            self.subj2sex = {}
            self.subj2age = {}

        self.nclass_list = [len(self.label_names)]
        self.expert_embeds = [0 for _ in range(len(self))]
        print("Data num", len(self), "Label name", self.label_names)
            
    def get_label(self, fn):
        labeln = fn.split('_')[2]
        if 'monkey' in labeln: labeln = 'task-monkey'
        return self.label_names.index(labeln), labeln
        
        
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, index):
        # label = self.labels[index]
        # data = (features - torch.mean(features, axis=0, keepdims=True)) / torch.std(features, axis=0, keepdims=True)
        if self.cached_data[index] is None:
            features = np.loadtxt(self.data_path[index], delimiter='\t') # N x T
            features = torch.from_numpy(features).float()
            x = torch.nan_to_num(features)
            x = torch.corrcoef(x)
            subjn = self.subject[index]
            eeg = np.load(self.eeg_path[index])
            eeg = torch.from_numpy(eeg).float()
            eeg = torch.nan_to_num(eeg)
        
            x[x.isnan()] = 0
            x[x.isinf()] = 0
            data = {
                # 'edge_index': edge_index,
                'eeg': eeg,
                'x': x,
                'y': self.labels[index],
                'sex': self.subj2sex[subjn] if subjn in self.subj2sex else -1,
                'age': self.subj2age[subjn] if subjn in self.subj2age else torch.tensor([-1]).float(),
                # 'edge_index_fc': edge_index_fc,
                # 'edge_index_sc': edge_index_sc
            }
            
            self.cached_data[index] = data
        data = self.cached_data[index]
        for t, tn in zip(self.transform, self.transform_tag):
            data[tn] = t(data)
        # data['expert_embed'] = self.expert_embeds[index]
        if self.expert_embeds[index] != 0:
            data['expert_embed'] = self.expert_embeds[index]
        if self.label_remap is not None:
            if data['y'] in self.label_remap:
                data['y'] = self.label_remap[data['y']]
        return data


DATASET_CLASS = {
    'adni': NeuroNetworkDataset,
    'oasis': NeuroNetworkDataset,
    'hcpa': NeuroNetworkDataset,
    'ukb': NeuroNetworkDataset,
    'hcpya': NeuroNetworkDataset,
    'ppmi': Dataset_PPMI_ABIDE,
    'abide': Dataset_PPMI_ABIDE,
    'neurocon': Dataset_PPMI_ABIDE,
    'taowu': Dataset_PPMI_ABIDE,
    'sz-diana': Dataset_SZ,
    'fmrieeg': Dataset_FMRIEEG,
}
np.random.seed(142857)
def cv_dataloader_generator(batch_size=4, num_workers=8, nfold=0, total_fold={'adni': 5,'hcpa': 5,'hcpya': 5,'abide': 10,'ppmi': 10,'taowu': 10,'neurocon': 10, 'sz-diana': 10}, few_shot=1, dataset=None, **kargs):
    kf = KFold(n_splits=total_fold[kargs['dname']], shuffle=True, random_state=142857)
    if dataset is None:
        dataset = DATASET_CLASS[kargs['dname']](**kargs)
    all_subjects = dataset.data_subj
    # labels = np.array([data['y'] for data in dataset.cached_data])
    train_index, index = list(kf.split(all_subjects))[nfold]
    ## Few shot
    # train_shots = [train_index[labels[train_index]==yi] for yi in np.unique(labels)]
    # train_index = []
    # for i in range(len(train_shots)):
    #     np.random.shuffle(train_shots[i])
    #     train_num = max(int(len(train_shots[i])*few_shot/len(train_shots)), 1) # at least one data
    #     train_index.extend(list(train_shots[i][:train_num]))
    ## Few shot done
    train_subjects = [all_subjects[i] for i in train_index]
    subjects = [all_subjects[i] for i in index]
    # Filter dataset based on training and validation subjects
    train_data = [di for di, subj in enumerate(dataset.subject) if subj in train_subjects]
    data = [di for di, subj in enumerate(dataset.subject) if subj in subjects]
    print(f'Fold {nfold + 1}, Train {len(train_subjects)} subjects, Val {len(subjects)} subjects, len(train_data)={len(train_data)}, len(data)={len(data)}')
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    valid_dataset = torch.utils.data.Subset(dataset, data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, loader, dataset


def cv_dataloader_generator_precompute_expert(batch_size=4, num_workers=8, nfold=0, total_fold={'adni': 5,'hcpa': 5,'hcpya': 5,'abide': 10,'ppmi': 10,'taowu': 10,'neurocon': 10, 'sz-diana': 10, 'fmrieeg': 5}, dataset=None, experts=[], expert_tags=[], device='cpu', **kargs):
    kf = KFold(n_splits=total_fold[kargs['dname']], shuffle=True, random_state=142857)
    # if os.path.exists(f'data/{""}')
    if dataset is None:
        dataset = DATASET_CLASS[kargs['dname']](**kargs)
        data_index = 0
        dloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        for data in tqdm(dloader, desc='Precompute Expert Embedding'):
            expert_embed = []
            for experti in range(len(experts)):
                if expert_tags[experti].split('_')[0] in ['BrainMass', 'BrainJEPA']:
                    x = data[expert_tags[experti].split('_')[0]][0]
                elif expert_tags[experti].split('_')[0] in ['EEG']:
                    x = data['eeg']
                else:
                    x = data['x']
                    
                with torch.no_grad():
                    y = experts[experti](x.to(device))
                    if isinstance(y, dict): y=y['hidden_state'][:,0]
                    expert_embed.append(y.detach().cpu()) # B x C
            # expert_embed = torch.stack(expert_embed, 1).detach().cpu()
            for bi in range(len(expert_embed[0])):
                dataset.expert_embeds[data_index+bi] = [e[bi] for e in expert_embed]
            data_index += len(expert_embed[0])
    all_subjects = dataset.data_subj
    # labels = np.array([data['y'] for data in dataset.cached_data])
    train_index, index = list(kf.split(all_subjects))[nfold]
    ## Few shot
    # train_shots = [train_index[labels[train_index]==yi] for yi in np.unique(labels)]
    # train_index = []
    # for i in range(len(train_shots)):
    #     np.random.shuffle(train_shots[i])
    #     train_num = max(int(len(train_shots[i])*few_shot/len(train_shots)), 1) # at least one data
    #     train_index.extend(list(train_shots[i][:train_num]))
    ## Few shot done
    train_subjects = [all_subjects[i] for i in train_index]
    subjects = [all_subjects[i] for i in index]
    # Filter dataset based on training and validation subjects
    train_data = [di for di, subj in enumerate(dataset.subject) if subj in train_subjects]
    data = [di for di, subj in enumerate(dataset.subject) if subj in subjects]
    print(f'Fold {nfold + 1}, Train {len(train_subjects)} subjects, Val {len(subjects)} subjects, len(train_data)={len(train_data)}, len(data)={len(data)}')
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    valid_dataset = torch.utils.data.Subset(dataset, data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, loader, dataset


if __name__ == '__main__':
    PretrainDataset()