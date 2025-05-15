import os
import matplotlib.pyplot as plt
import numpy as np
tgt = 'y'
root = '.'
data = {'k': [], 'F1': [], 'F1_std': [], 'dataset': []}
k_2_perc = {1: 1, 9: 25, 18: 50, 36: 100}
for fn in os.listdir(root):
    logfn = f'{root}/{fn}'
    if '3type' not in fn: continue
    if ('topk' not in fn or not fn.endswith('.log')) and not fn.endswith('onlyAAL.log'):continue
    # if 'hcp' in fn: continue
    if 'sex' in fn and tgt != 'sex': continue
    if tgt == 'sex' and 'sex' not in fn: continue
    if fn.endswith('onlyAAL.log'): 
        k = 36
    else:
        k = fn.split('_')[-1].replace('.log', '').replace('topk', '')
    if int(k) not in k_2_perc: continue
    k = k_2_perc[int(k)]
    with open(logfn, 'r') as logf:
        lines = logf.read().split('\n')
    if len(lines) <= 5: 
        print(logfn)
        continue
    line = lines[-5]
    print(fn.split('_')[-1].replace('.log', ''),line)
    if 'Mean F1 Score' not in line: 
        print(logfn)
        continue
    avg = line.split(', ')[0].split(' ')[-1]
    avg = float(avg) * 100
    std = line.split(', ')[1].split(' ')[-1]
    std = float(std) * 100
    if k == 25 : continue
    data['k'].append(k)
    data['F1'].append(avg)
    data['F1_std'].append(std)
    data['dataset'].append(fn.split('_')[2].replace('-sex', ''))
print(data)

cmap = plt.get_cmap('Set2')
plt.figure(figsize=(3,5))
skip_dn = ['neurocon', 'taowu', 'sz-diana', 'adni']
for k in data: data[k] = np.array(data[k])
dni = 0
ii = 0
shift = 2
shift_width = shift*len(np.unique(data['dataset']))
for dn in np.unique(data['dataset']):
    ind = data['dataset']==dn
    x = data['k'][ind] + dni - shift_width/2
    y = data['F1'][ind]
    y_err = data['F1_std'][ind]#/10
    y = y[np.argsort(x)]
    y_err = y_err[np.argsort(x)]
    x = x[np.argsort(x)]
    print(dn, 1, f'{y[0]:.2f}$'+'_{\pm'+f'{y_err[0]:.2f}'+'}$')
    if dn in skip_dn: continue
    eb = plt.errorbar(x, y, yerr=y_err, label=dn, marker='s', color=cmap(ii))
    eb[-1][0].set_linestyle('-.')
    ii += 1
    dni += shift
exit()
x = {'y': np.array([1, 1, 1, 1]), 'sex': np.array([1, 1, 1, 1])}
y = {'y': np.array([59.54, 50.68, 36.25, 38.69]), 'sex': np.array([40.32, 40.20, 78.08, 46.23])}
dns = np.array(['hcpa', 'hcpya', 'abide', 'ppmi'])
dni = 0
for dn in np.unique(data['dataset']):
    ind = dns==dn
    if dn in skip_dn: continue
    plt.scatter(x[tgt][ind]+3, y[tgt][ind], marker='*', s=200, color=cmap(dni))
    dni += 1

plt.legend()
# plt.xticks([1, 25, 50, 100], labels=['1', '25%', '50%', '100%'])
plt.xticks([1, 50, 100], labels=['1', '50%', '100%'])
ylim = {'y': [50,100], 'sex': [60,90]}
plt.ylim(ylim[tgt])
plt.tight_layout()

plt.savefig(f'topk_moe_allEx_f1_{tgt}.png')
plt.savefig(f'topk_moe_allEx_f1_{tgt}.svg')
plt.close()
    
