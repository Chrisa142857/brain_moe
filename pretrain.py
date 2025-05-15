from datasets import dataloader_generator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from models import BrainMoE, BrainMoEDecoder
# from models.heads import Classifier, BNDecoder
# from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os
import numpy as np
from datetime import datetime

MODEL_BANK = {
    'brainMoE': BrainMoE,
    'brainMoEDecoder': BrainMoEDecoder,
    # 'neurodetourSingleSC': neuro_detour.DetourTransformerSingleSC,
    # 'bnt': brain_net_transformer.BrainNetworkTransformer,
    # 'braingnn': brain_gnn.Network,
    # 'bolt': bolt.get_BolT,
    # 'graphormer': graphormer.Graphormer,
    # 'nagphormer': nagphormer.TransformerModel,
    # 'transformer': vanilla_model.Transformer,
    # 'gcn': vanilla_model.GCN,
    # 'sage': vanilla_model.SAGE,
    # 'sgc': vanilla_model.SGC,
    # 'none': brain_identity.Identity
}

ATLAS_ROI_N = {
    'AAL_116': 116,
    'Gordon_333': 333,
    'Shaefer_100': 100,
    'Shaefer_200': 200,
    'Shaefer_400': 400,
}

# LOSS_FUNCS = {
#     'y': nn.CrossEntropyLoss(),
#     'sex': nn.CrossEntropyLoss(),
#     'age': nn.MSELoss(), 
# }
# LOSS_W = {
#     'y': 1,
#     'sex': 1,
#     'age': 1e-4,
# }

def main():
    parser = argparse.ArgumentParser(description='BrainMoE')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default = 200)
    parser.add_argument('--model', type=str, default = 'brainMoE')
    parser.add_argument('--max_patience', type=int, default = 50)
    parser.add_argument('--hiddim', type=int, default = 1024)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--atlas', nargs='+', default = ['AAL_116','Gordon_333','Shaefer_400'], required=False)
    parser.add_argument('--classifier_aggr', type=str, default = 'learn')
    parser.add_argument('--savemodel', action='store_true')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--device', type=str, default = 'cuda:0')
    parser.add_argument('--decoder_layer', type=int, default = 4)
    parser.add_argument('--nhead', type=int, default = 4)
    parser.add_argument('--tgt_atlas', type=str, default = 'AAL_116')
    # parser.add_argument('--isolate_expert', action='store_true')

    args = parser.parse_args()
    args.savemodel = True
    print(args)
    expdate = str(datetime.now())
    expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    device = args.device
    # hiddim = args.hiddim
    # nclass = DATA_CLASS_N[args.dataname]
    # if args.force_2class:
    #     nclass = 2
    # dataset = None
    # Initialize lists to store evaluation metrics
    accuracies_dict = {}
    f1_scores_dict = {}
    prec_scores_dict = {}
    rec_scores_dict = {}
    auc_scores_dict = {}
    # node_sz = ATLAS_ROI_N[args.atlas]
    # if args.models != 'neurodetour':
    # transform = None
    # dek, pek = 0, 0
    # if args.node_attr != 'BOLD':
    #     input_dim = node_sz
    # else:
    #     input_dim = args.bold_winsize
    # transform = DATA_TRANSFORM[args.models]
    # testset = args.testname


    # dataloaders = [dataloader_generator(batch_size=args.batch_size, tgt_atlas_name=atlas) for atlas in args.atlas]  
    # datasets = [dl[-1] for dl in dataloaders] 
    # expert_tags = []
    # for dset in datasets:
    #     expert_tags += list(np.unique(dset.expert_tags))
    train_loader, val_loader, dset = dataloader_generator(batch_size=args.batch_size, tgt_atlas_name=args.tgt_atlas)
    expert_tags = list(np.unique(dset.expert_tags))
    experts = MODEL_BANK[args.model](expert_tag=expert_tags, nlayer=args.decoder_layer, head_num=args.nhead, hid_dim=args.hiddim).to(device)
    # experts = nn.ModuleList([
    #     BrainExpert(input_sz=int(t.split('_')[-1]), nlayer=args.decoder_layer, head_num=args.nhead, hid_dim=args.hiddim)
    #     for t in expert_tags
    # ]).to(device)
    params = list(experts.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay) 
    param_amount = sum([np.cumprod(p.shape)[-1] for p in params]) / 1e+6
    param_amount = f'{param_amount:.2f}M'
    print('expert_tags:', expert_tags)
    print('param_amount:', param_amount)
    if args.savemodel:
        mweight_fn = f'model_weights/{args.model}_param{param_amount}_{"-".join(args.atlas)}_{expdate}'
        os.makedirs(mweight_fn, exist_ok=True)

    # optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
    # print(optimizer)
    # best_f1 = 0
    # patience = 0
    best_f1 = {}
    best_acc = {}
    best_prec = {}
    best_rec = {}
    best_auc = {}
    for epoch in range(1, args.epochs+1):
        # print(datetime.now(), 'train start')
        # for train_loader, val_loader, _ in dataloaders:
        train(experts, device, train_loader, optimizer, args)
        # print(datetime.now(), 'train done, test start')
        # acc, prec, rec, f1 = eval(model, classifier, device, val_loader)
        scores = eval(experts, device, val_loader, args)
        # print(datetime.now(), 'test done')
        log = f'{datetime.now()}, Epoch {epoch:03d} eval [Accuracy, F1 Score]:'
        for k in scores:
            acc, prec, rec, f1, auc_score = scores[k]
            log += f'({k}) [{acc:.6f},  {f1:.6f}], \t'
            if k not in best_f1:
                best_f1[k] = -torch.inf
                best_acc[k] = -torch.inf
                best_prec[k] = -torch.inf
                best_rec[k] = -torch.inf
                best_auc[k] = -torch.inf
            
            if f1 > best_f1[k]:
            # if f1 + auc_score >= best_auc[k] + best_f1[k]:
                best_f1[k] = f1
                best_acc[k] = acc
                best_prec[k] = prec
                best_rec[k] = rec 
                best_auc[k] = auc_score
                # patience = 0
                    
                if args.savemodel:
                    for experti, t in enumerate(experts.expert_tag):
                        if k not in t: continue
                        torch.save(experts.experts[experti].state_dict(), f'{mweight_fn}/BrainExpert_{t}_best.pt')
            # else:
                # patience += 1
        print(log)
        # if patience > args.max_patience: break


    log = f'Pretrain Done [Accuracy, F1 Score, Prec, Rec, AUC]:'
    for k in best_acc:
        if k not in accuracies_dict:
            accuracies_dict[k] = []
            f1_scores_dict[k] = []
            prec_scores_dict[k] = []
            rec_scores_dict[k] = []
            auc_scores_dict[k] = []
        accuracies_dict[k].append(best_acc[k])
        f1_scores_dict[k].append(best_f1[k])
        prec_scores_dict[k].append(best_prec[k])
        rec_scores_dict[k].append(best_rec[k])
        auc_scores_dict[k].append(best_auc[k])
        log += f'({k}) [{best_acc[k]}, {best_f1[k]}, {best_prec[k]}, {best_rec[k]}, {best_auc[k]}], \t'
    print(log)


def train(model, device, loader, optimizer, args):
    model.train()
    train_tgt = 'y'
    losses = []
    # y_true_dict = {k: [] for k in model.expert_tag}
    # y_scores_dict = {k: [] for k in model.expert_tag}
    # loss_fn = nn.MSELoss()
    y_true_dict = {}
    y_scores_dict = {}
    loss_fn = nn.CrossEntropyLoss()
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    log_interval = 50
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch['x'] = batch['x'].to(device)
        outputs = model(batch['x'], loader.dataset.dataset.expert_tags)
        if args.model == 'brainMoE':
            output = torch.cat([output[train_tgt] for output in outputs], 1)
            loss = loss_fn(output, batch[train_tgt].argmax(1).to(device))
            atlas_name = '_'.join(outputs[0]['expert_tag'].split('_')[1:])
            if atlas_name not in y_true_dict:
                y_true_dict[atlas_name] = []
                y_scores_dict[atlas_name] = []
            y_true_dict[atlas_name].append(batch[train_tgt].argmax(1))
            y_scores_dict[atlas_name].append(output.detach().cpu().argmax(1))
        elif args.model == 'brainMoEDecoder':
            loss = 0
            for output in outputs:
                loss += loss_fn(output[train_tgt], batch[train_tgt].argmax(1).to(device))
                atlas_name = output['expert_tag']
                if atlas_name not in y_true_dict:
                    y_true_dict[atlas_name] = []
                    y_scores_dict[atlas_name] = []
                y_true_dict[atlas_name].append(batch[train_tgt].argmax(1))
                y_scores_dict[atlas_name].append(output[train_tgt].detach().cpu().argmax(1))
            loss /= len(outputs)
            
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        if step % log_interval == 0:
            print(f'{datetime.now()} [{step:03d}/{len(loader)}] loss: {loss:.5f}')
    
    logs = [f'Train loss: {np.mean(losses):.6f}']
    for k in y_true_dict:
        y_true = torch.cat(y_true_dict[k], dim = 0).detach().cpu().numpy()
        y_scores = torch.cat(y_scores_dict[k], dim = 0).detach().cpu().numpy()
        # y_true = y_true.numpy()
        # y_scores = (y_scores.numpy() >= 0.5).astype(np.int32)
        acc = accuracy_score(y_true, y_scores)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
        logs.append(f'Expert {k} \t Acc: {acc:.6f} \t F1: {f1:.6f}')
    
    print('\n'.join(logs))

def eval(model, device, loader, args):
    model.eval()
    train_tgt = 'y'
    losses = []
    # y_true_dict = {k: [] for k in model.expert_tag}
    # y_scores_dict = {k: [] for k in model.expert_tag}
    # loss_fn = nn.MSELoss()
    y_true_dict = {}
    y_scores_dict = {}
    loss_fn = nn.CrossEntropyLoss()
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch['x'] = batch['x'].to(device)
        with torch.no_grad():
            outputs = model(batch['x'], loader.dataset.dataset.expert_tags)
            
        if args.model == 'brainMoE':
            output = torch.cat([output[train_tgt] for output in outputs], 1)
            loss = loss_fn(output, batch[train_tgt].argmax(1).to(device))
            atlas_name = '_'.join(outputs[0]['expert_tag'].split('_')[1:])
            if atlas_name not in y_true_dict:
                y_true_dict[atlas_name] = []
                y_scores_dict[atlas_name] = []
            y_true_dict[atlas_name].append(batch[train_tgt].argmax(1))
            y_scores_dict[atlas_name].append(output.detach().cpu())
        elif args.model == 'brainMoEDecoder':
            loss = 0
            for output in outputs:
                loss += loss_fn(output[train_tgt], batch[train_tgt].argmax(1).to(device))
                atlas_name = output['expert_tag']
                if atlas_name not in y_true_dict:
                    y_true_dict[atlas_name] = []
                    y_scores_dict[atlas_name] = []
                y_true_dict[atlas_name].append(batch[train_tgt].argmax(1))
                y_scores_dict[atlas_name].append(output[train_tgt].detach().cpu())
            loss /= len(outputs)
        losses.append(loss.detach().cpu().item())
    
    logs = [f'Eval loss: {np.mean(losses):.6f}']
    for k in y_true_dict:
        y_true = torch.cat(y_true_dict[k], dim = 0).detach().cpu().numpy()
        y_scores = torch.cat(y_scores_dict[k], dim = 0).detach().cpu().numpy().argmax(1)
        # y_true = y_true.numpy()
        # y_scores = y_scores.numpy().argmax(1)
        acc = accuracy_score(y_true, y_scores)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
        logs.append(f'Expert {k} \t Acc: {acc:.6f} \t F1: {f1:.6f}')
    
    print('\n'.join(logs))
    scores = {}    
    for k in y_true_dict:
        y_true = torch.cat(y_true_dict[k], dim = 0).detach().cpu().numpy()
        y_pred = torch.cat(y_scores_dict[k], dim = 0).detach().cpu().numpy()
        y_scores = y_pred.argmax(1)
        # y_true = y_true.numpy()
        # y_scores = y_scores.numpy()
        acc = accuracy_score(y_true, y_scores)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
        # fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1], pos_label=1)
        # auc_score = auc(fpr, tpr)
        auc_score = 0
        for y in np.unique(y_true):
            fpr, tpr, thresholds = roc_curve((y_true==y).astype(np.float32), y_pred[:, y.astype(np.int32)], pos_label=1)
            auc_score += auc(fpr, tpr)
        auc_score /= len(np.unique(y_true))
        scores[k] = [acc, prec, rec, f1, auc_score]

    return scores

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()