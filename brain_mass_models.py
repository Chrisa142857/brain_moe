# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch.nn as nn
import torch
from torch.nn import TransformerEncoderLayer
import numpy as np
import random
from nilearn.connectome import ConnectivityMeasure


class BNTF(nn.Module):
    def __init__(self,feature_dim,depth,heads,dim_feedforward,node_num):
        super().__init__()
        # self.num_patches = 100

        self.attention_list = nn.ModuleList()
        self.node_num = node_num
        for _ in range(int(depth)):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=self.node_num, nhead=int(heads), dim_feedforward=dim_feedforward, 
                                        batch_first=True)
            )
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.node_num, 8),
            nn.LeakyReLU()
        )
        self.hid_dim = 960

        final_dim = 8 * self.node_num

        self.g = MLPHead(final_dim, final_dim * 2, feature_dim)
        
    def forward(self,img):
        bz, _, _, = img.shape

        for atten in self.attention_list:
            img = atten(img)

        node_feature = self.dim_reduction(img)
        node_feature = node_feature.reshape((bz, -1))
        # node_feature = self.g(node_feature)
        return node_feature


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


def mask_timeseries(timeser, mask = 30):
    rnd = np.random.random()
    time_len = timeser.shape[1]
    mask_index = np.array(random.sample(list(np.arange(0,time_len)),mask))
    bool_mask = np.zeros((time_len))
    bool_mask[mask_index]=1
    bool_mask = bool_mask.astype(bool)

    return timeser[:,~bool_mask]
    
def mask_timeseries_per(timeser, mask = 30):
    rnd = np.random.random()

    time_len = timeser.shape[1]
    mask_len = int(mask * time_len /100)
    mask_index = np.array(random.sample(list(np.arange(0,time_len)),mask_len))
    bool_mask = np.zeros((time_len))
    bool_mask[mask_index]=1
    bool_mask = bool_mask.astype(bool)

    return timeser[:,~bool_mask]

def random_timeseries(timeser,sample_len):
    time_len = timeser.shape[1]
    st_thres = 1
    if time_len <= sample_len + st_thres:
        return timeser

    select_range = time_len - sample_len
    if select_range < 1:
        return timeser

    st = random.sample(list(np.arange(st_thres,select_range)),1)[0]
    return timeser[:,st:st+sample_len]

class EvalTransformation:
    def __init__(self, classn=2,mask_way='mask',mask_len=10, time_len=30,target='y',is_train = True, is_test = False, roi_num=120, return_sex=False, return_dict=False):
        # self.template = 'sch'
        self.classn = classn
        self.is_test = is_test
        self.is_train = is_train
        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len
        self.roi_num = roi_num
        self.target = target
        self.return_sex = return_sex
        self.return_dict = return_dict
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

    def __call__(self,data):
        img = data['x'].numpy()[:self.roi_num] # N x T (one brain)
        lbl = data[self.target]#.item()
        if self.is_train is True:
            if self.mask_way == 'mask':
                slices = [mask_timeseries(img,self.mask_len).T]
            elif self.mask_way =='random':
                slices = [random_timeseries(img,self.time_len).T]
            elif self.mask_way =='mask_per':
                slices = [mask_timeseries_per(img,mask=self.mask_len).T]
            else:
                slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        elif self.is_test is False:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices)[0]
        else:
            # slices = [img.T]
            slices = [mask_timeseries_per(img,mask=self.mask_len).T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        if self.target != 'age':
            onehot_lbl = np.zeros((self.classn))
            onehot_lbl[lbl] = 1
        else:
            onehot_lbl = np.zeros((101))
            onehot_lbl[int(lbl)] = 1
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        if self.roi_num > correlation_matrix.shape[1]:
            correlation_matrix = np.concatenate([correlation_matrix, np.zeros([correlation_matrix.shape[1], 4])], 1)
            correlation_matrix = np.concatenate([correlation_matrix, np.zeros([4, correlation_matrix.shape[1]])], 0)
        
        # if self.return_dict:
        #     return {
        #         'x': correlation_matrix,
        #         self.target: lbl
        #     }
            
        if not self.return_sex:
            return correlation_matrix.astype(np.float32),onehot_lbl

        return correlation_matrix.astype(np.float32),onehot_lbl,data['sex'],data['age']
    