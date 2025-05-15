import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# valid_dn = ['taowu', 'adni', 'ppmi', 'ppmi2cls', 'abide', 'neurocon', 'hcpa', 'hcpya', 'sz-diana']
# data = {'backbone': [], 'classifier': [], 'data': [], 'data type': [], 'acc': [], 'f1': []}
# data = {'backbone': [], 'classifier': [], 'data': [], 'acc': [], 'f1': []}
data = {'model': [], 'data': [], 'acc_avg': [], 'f1_avg': [], 'acc_std': [], 'f1_std': [], 'acc_str': [], 'f1_str': []}
logtag = '.'
tgt_task = 'finetune'
for logf in os.listdir(logtag):
    if not logf.endswith('.log'): continue
    if 'mlp1Mix' in logf: continue
    with open(f'{logtag}/{logf}', 'r') as f:
        lines = f.read().split('\n')[-6:]
    if 'Mean' not in lines[0]: continue
    task = logf.split('_')[0]
    if task != tgt_task: continue
    mn = logf.split('_')[1] + '_' + logf.split('_')[-1].split('.')[0]
    dn = logf.split('_')[2]
    # if dn not in valid_dn: continue
    if len(lines[0].split('Accuracy: ')) < 3: continue
    acc_avg = float(lines[0].split('Accuracy: ')[1].replace(', Std ', ''))
    acc_std = float(lines[0].split('Accuracy: ')[2])
    f1_avg = float(lines[1].split('F1 Score: ')[1].replace(', Std ', ''))
    f1_std = float(lines[1].split('F1 Score: ')[2])
    data['model'].append(mn)
    data['data'].append(dn)
    # data['data type'].append(dt)
    # data['acc'].append(f"{(acc_avg*100):.5f}+-{(acc_std*100):.5f}")
    # data['f1'].append(f"{(f1_avg*100):.5f}+-{(f1_std*100):.5f}")

    data['acc_avg'].append(acc_avg*100)
    data['f1_avg'].append(f1_avg*100)
    data['acc_std'].append(acc_std*100)
    data['f1_std'].append(f1_std*100)
    data['f1_str'].append(f'{f1_avg*100:.2f}' + '$_{\pm'+ f'{f1_std*100:.2f}' + '}$')
    data['acc_str'].append(f'{acc_avg*100:.2f}' + '$_{\pm'+ f'{acc_std*100:.2f}' + '}$')

data = pd.DataFrame(data)
# one = data[data['backbone']=='none'][data['classifier']=='decoder32']
# x = one[one['data']=='adni']
# acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
# f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
# ad = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#         f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
# x = one[one.data.isin(['ppmi', 'taowu', 'neurocon'])]
# acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
# f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
# pd = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#         f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
# x = one[one['data']=='abide']
# acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
# f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
# aut = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#         f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
# print(' & '.join(ad + pd + aut))
# # exit()
# # interested_bb = ['braingnn', 'bnt', 'bolt', 'graphormer', 'nagphormer', 'neurodetour']
# interested_bb = ['braingnn', 'bnt', 'bolt', 'graphormer', 'nagphormer', 'neurodetour']
# for bb in interested_bb:
#     one = data[data['backbone']==bb]
#     print(bb)
#     x = one[one['data']=='adni']
#     acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
#     f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
#     ad = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#           f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
#     x = one[one.data.isin(['ppmi', 'taowu', 'neurocon'])]
#     acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
#     f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
#     pd = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#           f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
#     x = one[one['data']=='abide']
#     acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
#     f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
#     aut = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#           f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
#     print(' & '.join(ad + pd + aut))

data = data.sort_values('model')
data = data.sort_values('data')
# data = data.sort_values('data type')
results = '\n'
for unid in data['data'].unique():
    df = data[data['data']==unid]
    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    df.drop("data",axis=1,inplace=True)
    results += f'\n\n## Data: {unid} \n\n'
    results += df.to_markdown()
    results += '\n'

with open(f'brainmoe_in_markdown.md', 'w') as f:
    f.write(results)
