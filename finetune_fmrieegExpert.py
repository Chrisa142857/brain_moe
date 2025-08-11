

from datasets import cv_dataloader_generator_precompute_expert
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from models import RouterMultiModel, BrainMoE, BrainMoEDecoder#brain_net_transformer, neuro_detour, brain_gnn, brain_identity, bolt, graphormer, nagphormer, vanilla_model
from brain_jepa_models import VisionTransformerEval as BrainJEPA_expert
from brain_jepa_models import EvalTransformation as BrainJEPA_tform
from brain_mass_models import BNTF as BrainMass_expert
from brain_mass_models import EvalTransformation as BrainMass_tform
from cbramod_models import CBraMod
from einops.layers.torch import Rearrange

# from models.heads import Classifier, BNDecoder
# from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os, yaml
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

MODEL_BANK = {
    'brainMoE': BrainMoE,
    'brainMoEDecoder': BrainMoEDecoder,
}
# CLASSIFIER_BANK = {
#     'mlp': nn.Linear,
#     'gcn': GCNConv,
#     'gat': GATConv,
#     'sage': SAGEConv,
#     'sgc': SGConv
# }
# DATA_TRANSFORM = {
#     'neurodetour': None,
#     'neurodetourSingleFC': None,
#     'neurodetourSingleSC': None,
#     'bnt': None,
#     'braingnn': None,
#     'bolt': None,
#     'graphormer': graphormer.ShortestDistance(),
#     'nagphormer': nagphormer.NAGdataTransform(),
#     'transformer': None,
#     'gcn': None,
#     'sage': None,
#     'sgc': None,
#     'none': None
# }
ATLAS_ROI_N = {
    'AAL_116': 116,
    'Gordon_333': 333,
    'Shaefer_100': 100,
    'Shaefer_200': 200,
    'Shaefer_400': 400,
    'D_160': 160
}
DATA_CLASS_N = {
    'ukb': 2,
    'hcpa': 4,
    'hcpya': 7,
    'adni': 2,
    'oasis': 2,
    'oasis': 2,
    'ppmi': 4,
    'abide': 2,
    'neurocon': 2,
    'taowu': 2,
    'sz-diana': 2,
    'fmrieeg': 8,
}
LOSS_FUNCS = {
    'y': nn.CrossEntropyLoss(),
    'sex': nn.CrossEntropyLoss(),
    'age': nn.MSELoss(), 
}
LOSS_W = {
    'y': 1,
    'sex': 1,
    'age': 1e-4,
}

torch.manual_seed(42)

def main():
    parser = argparse.ArgumentParser(description='BrainMoE')
    
    parser.add_argument('--dataname', type=str, default = 'fmrieeg')
    parser.add_argument('--nlayer', type=int, default = 32)
    parser.add_argument('--nhead', type=int, default = 8)
    parser.add_argument('--device', type=str, default = 'cuda:3')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default = 200)
    parser.add_argument('--model', type=str, default = 'brainMoE')
    parser.add_argument('--max_patience', type=int, default = 50)
    parser.add_argument('--hiddim', type=int, default = 1024)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--expert_atlas', nargs='+', default = ['Shaefer_400'], required=False)
    parser.add_argument('--atlas', nargs='+', default = 'Shaefer_400', required=False)
    parser.add_argument('--savemodel', action='store_true')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--decoder_layer', type=int, default = 4)
    parser.add_argument('--decoder_head', type=int, default = 4)
    parser.add_argument('--decoder_hiddim', type=int, default = 1024)
    parser.add_argument('--train_obj', type=str, default = 'age')
    parser.add_argument('--few_shot', type=float, default = 1)
    parser.add_argument('--force_2class', action='store_true')
    parser.add_argument('--nofreeze_expert', action='store_true')
    parser.add_argument('--use_pseudolabel', action='store_true')
    parser.add_argument('--single_expert', action='store_true')
    parser.add_argument('--nofmri', action='store_true')
    parser.add_argument('--noeeg', action='store_true')
    parser.add_argument('--topk_expert', type=int, default = -1)
    args = parser.parse_args()
    print(args)
    expdate = str(datetime.now())
    expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    device = args.device
    hiddim = args.hiddim
    target = args.train_obj
    nclass = DATA_CLASS_N[args.dataname]
    if args.force_2class or target == 'sex':
        nclass = 2
    if target == 'age':
        nclass = 1
    dataset = None
    # Initialize lists to store evaluation metrics
    accuracies_dict = {}
    f1_scores_dict = {}
    prec_scores_dict = {}
    rec_scores_dict = {}
    auc_scores_dict = {}
    # taccuracies = []
    # tf1_scores = []
    # tprec_scores = []
    # trec_scores = []
    node_sz = ATLAS_ROI_N[args.atlas]
    # if args.models != 'neurodetour':
    # transform = None
    # dek, pek = 0, 0
    # if args.node_attr != 'BOLD':
    #     input_dim = node_sz
    # else:
    #     input_dim = args.bold_winsize
    # transform = DATA_TRANSFORM[args.models]
    # testset = args.testname
    # if args.savemodel:
    #     mweight_fn = f'model_weights/{args.models}_{args.atlas}_boldwin{args.bold_winsize}_{args.adj_type}{args.node_attr}'
    #     os.makedirs(mweight_fn, exist_ok=True)
    _nfold = {
        'ukb': 5,
        'hcpa': 5,
        'hcpya': 5,
        'adni': 5,
        'oasis': 5,
        'ppmi': 10,
        'abide': 10,
        'neurocon': 10,
        'taowu': 10,
        'sz-diana': 10,
        'fmrieeg': 5,
    }

    weight_dir = 'model_weights'
    expert_tags = []
    experts = []
    if not args.nofmri:
        weight_path = []
        for expert_dir in os.listdir(weight_dir):
            if f'{args.model}_param' not in expert_dir: continue
            expert_fns = os.listdir(f'{weight_dir}/{expert_dir}')
            expert_fns = [fn for fn in expert_fns if '_'.join(fn.split('_')[-3:-1]) in args.expert_atlas]
            if args.single_expert: expert_fns = [fn for fn in expert_fns if 'task-rest_AAL_116' in fn] # single expert only AAL_116
            if len(expert_fns) == 0: continue
            expert_tags += [fn.replace('BrainExpert_', '').replace('_best.pt', '') for fn in expert_fns]
            weight_path += [f'{weight_dir}/{expert_dir}/{fn}' for fn in expert_fns]
        print('Loading experts', expert_tags)
        classif_experts = MODEL_BANK[args.model](expert_tag=expert_tags, nlayer=args.decoder_layer, head_num=args.decoder_head, hid_dim=args.decoder_hiddim).to(device)
        for experti in range(len(classif_experts.experts)):
            classif_experts.experts[experti].load_state_dict(torch.load(weight_path[experti], map_location='cpu', weights_only=True))
        
        classif_experts.eval()
        for expert in classif_experts.experts:
            experts.append(expert)


    if not args.noeeg:
        chpt_path = '../CBraMod/pretrained_weights/pretrained_weights.pth' # download from https://github.com/wjq-learning/CBraMod
        expert = CBraMod()
        msg = expert.load_state_dict(torch.load(chpt_path, map_location='cpu'))
        expert.proj_out = Rearrange('b c s p -> b (c s p)')
        expert.hid_dim = 22*4*200
        print(msg)
        expert.to(device)
        expert.eval()
        experts.append(expert)
        expert_tags.append('EEG_CBraMod')


    print('Loaded experts', expert_tags)
    transform = []
    transform_name = []
    # brainMass_tform = BrainMass_tform(classn=nclass, mask_way=mask_way,mask_len=mask_len,time_len=time_len,target=target,is_train=True,is_test=False, roi_num=roi_num, return_dict=True)
    # transform.append(brainMass_tform)
    # transform_name.append('BrainMass')
    # brainjepa_tform = BrainJEPA_tform(target=target, crop_size=[116,160], use_normalization=True, downsample=True)
    # transform.append(brainjepa_tform)
    # transform_name.append('BrainJEPA')

    # foldi = 0
    # _foldi = 0
    # for foldi in range(5):
    # while foldi < 5 and _foldi < _nfold[args.dataname]:
    #     _foldi += 1
    for foldi in range(1, _nfold[args.dataname]):
        train_loader, val_loader, dataset = cv_dataloader_generator_precompute_expert(batch_size=args.batch_size, nfold=foldi, dataset=dataset, experts=experts, expert_tags=expert_tags, device=device,
                                                                 dname=args.dataname, atlas_name=args.atlas, transform=transform, transform_tag=transform_name)

        uni_label = torch.cat([data[args.train_obj] for data in val_loader]).unique()
        if args.force_2class: uni_label = uni_label[uni_label<=1]
        print(uni_label)
        if len(uni_label) == 1: continue
        # foldi += 1
            
        # model = MODEL_BANK[args.models](node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=args.batch_size, device=device, nlayer=args.nlayer, heads=args.nhead).to(device)
        # print(sum([p.numel() for p in model.parameters()]))
        # exit()
        # if not args.decoder:
        #     classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nlayer=args.decoder_layer, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), aggr=args.classifier_aggr).to(device)
        # else:
        #     classifier = BNDecoder(hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), nlayer=args.decoder_layer, head_num=8, return_intermediate=False).to(device)
        model = RouterMultiModel(experts=experts, expert_tags=expert_tags, target=args.train_obj, freeze_expert=not args.nofreeze_expert, use_pseudolabel=args.use_pseudolabel, device=device, input_sz=node_sz, nclass=nclass, nlayer=args.nlayer, head_num=args.nhead, hid_dim=hiddim).to(device)
        optimizer = optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.decay) 
        # optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # print(optimizer)
        # best_f1 = 0
        patience = 0
        best_f1 = {}
        best_acc = {}
        best_prec = {}
        best_rec = {}
        best_auc = {}
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            print(datetime.now(), 'train start')
            train(model, experts, device, train_loader, optimizer, args)
            print(datetime.now(), 'train done, test start')
            # acc, prec, rec, f1 = eval(model, classifier, device, val_loader)
            scores = eval(model, experts, device, val_loader, args)
            print(datetime.now(), 'test done')
            log = f'Dataset: {args.dataname} [Accuracy, F1 Score]:'
            for k in scores:
                acc, prec, rec, f1, auc_score = scores[k]
                if scores[k][0] == -1:
                    f1 = -1*f1

                log += f'({k}) [{acc:.6f},  {f1:.6f}], \t'
                if k not in best_f1:
                    best_f1[k] = -torch.inf
                    best_acc[k] = -torch.inf
                    best_prec[k] = -torch.inf
                    best_rec[k] = -torch.inf
                    best_auc[k] = -torch.inf
                
                if f1 >= best_f1[k]:
                # if f1 + auc_score >= best_auc[k] + best_f1[k]:
                    best_f1[k] = f1
                    best_acc[k] = acc
                    best_prec[k] = prec
                    best_rec[k] = rec 
                    best_auc[k] = auc_score
                    if k == args.train_obj:
                        patience = 0
                        
                    # if args.savemodel:
                    #     torch.save(model.state_dict(), f'{mweight_fn}/bb_fold{i}_{dname}Best-{k}_{expdate}.pt')
                    #     torch.save(classifier.state_dict(), f'{mweight_fn}/head_fold{i}_{dname}Best-{k}_{expdate}.pt')
                elif k == args.train_obj:
                    patience += 1
            print(log)
            if patience > args.max_patience: break

            # # pbar.set_description(f'Accuracy: {acc:.6f}, F1 Score: {f1:.6f}, Epoch')
            # if f1 >= best_f1:
            #     if f1 > best_f1: 
            #         patience = 0
            #     else:
            #         patience += 1
            #     best_f1 = f1
            #     best_acc = acc
            #     best_prec = prec
            #     best_rec = rec
            #     best_state = model.state_dict()
            #     best_cls_state = classifier.state_dict()
            #     if args.savemodel:
            #         torch.save(model.state_dict(), f'{mweight_fn}/fold{i}_{expdate}.pt')
            # else:
            #     patience += 1
            # if patience > args.max_patience: break
        
        # accuracies.append(best_acc)
        # f1_scores.append(best_f1)
        # prec_scores.append(best_prec)
        # rec_scores.append(best_rec)
        # print(f'Accuracy: {best_acc}, F1 Score: {best_f1}, Prec: {best_prec}, Rec: {best_rec}')
        # if args.testname != 'None':
        #     model.load_state_dict(best_state)
        #     classifier.load_state_dict(best_cls_state)
        #     tacc, tprec, trec, tf1 = eval(model, classifier, device, test_loader, hcpatoukb=args.testname in ['hcpa', 'ukb'])
        #     print(f'Testset: Accuracy: {tacc}, F1 Score: {tf1}, Prec: {tprec}, Rec: {trec}')
        #     taccuracies.append(tacc)
        #     tf1_scores.append(tprec)
        #     tprec_scores.append(trec)
        #     trec_scores.append(tf1)
        log = f'Dataset: {args.dataname} [Accuracy, F1 Score, Prec, Rec, AUC]:'
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

    # # Calculate mean and standard deviation of evaluation metrics
    for k in accuracies_dict:
        accuracies = accuracies_dict[k]
        f1_scores = f1_scores_dict[k]
        prec_scores = prec_scores_dict[k]
        rec_scores = rec_scores_dict[k]
        auc_scores = auc_scores_dict[k]
        mean_accuracy = sum(accuracies) / len(accuracies)
        std_accuracy = torch.std(torch.tensor(accuracies).float())
        mean_f1_score = sum(f1_scores) / len(f1_scores)
        std_f1_score = torch.std(torch.tensor(f1_scores).float())
        mean_prec_score = sum(prec_scores) / len(prec_scores)
        std_prec_score = torch.std(torch.tensor(prec_scores).float())
        mean_rec_score = sum(rec_scores) / len(rec_scores)
        std_rec_score = torch.std(torch.tensor(rec_scores).float())
        mean_auc_score = sum(auc_scores) / len(auc_scores)
        std_auc_score = torch.std(torch.tensor(auc_scores).float())
        print(f'Dataset: {args.dataname} ({k})')
        print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
        print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
        print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
        print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')
        print(f'Mean AUC: {mean_auc_score}, Std AUC: {std_auc_score}')

    # mean_accuracy = sum(accuracies) / len(accuracies)
    # std_accuracy = torch.std(torch.tensor(accuracies))
    # mean_f1_score = sum(f1_scores) / len(f1_scores)
    # std_f1_score = torch.std(torch.tensor(f1_scores))
    # mean_prec_score = sum(prec_scores) / len(prec_scores)
    # std_prec_score = torch.std(torch.tensor(prec_scores))
    # mean_rec_score = sum(rec_scores) / len(rec_scores)
    # std_rec_score = torch.std(torch.tensor(rec_scores))

    # print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
    # print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
    # print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
    # print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')

    # if args.testname != 'None':
    #     mean_accuracy = sum(taccuracies) / len(taccuracies)
    #     std_accuracy = torch.std(torch.tensor(taccuracies))
    #     mean_f1_score = sum(tf1_scores) / len(tf1_scores)
    #     std_f1_score = torch.std(torch.tensor(tf1_scores))
    #     mean_prec_score = sum(tprec_scores) / len(tprec_scores)
    #     std_prec_score = torch.std(torch.tensor(tprec_scores))
    #     mean_rec_score = sum(trec_scores) / len(trec_scores)
    #     std_rec_score = torch.std(torch.tensor(trec_scores))
    #     print(f'Test set: {args.testname}')
    #     print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
    #     print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
    #     print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
    #     print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')
        
def train(model, experts, device, loader, optimizer, args):
    model.train()
    losses = []
    y_true_dict = {}#{k: [] for k in LOSS_FUNCS}
    y_scores_dict = {}#{k: [] for k in LOSS_FUNCS}
    # loss_fn = nn.CrossEntropyLoss()
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        # batch = batch.to(device)
        y = model(batch['x'].to(device), [embed.to(device) for embed in batch['expert_embed']], topk=args.topk_expert)
        # if force_2class:
        #     _2cls_ind = torch.where(batch['y']<=1)[0]
        #     y['y'] = y['y'][_2cls_ind]
        #     batch['y'] = batch['y'][_2cls_ind]
        loss = 0
        for k in y:
            loss += LOSS_W[k]*LOSS_FUNCS[k](y[k][batch[k] != -1], batch[k][batch[k] != -1].to(device))
        if hasattr(model, 'loss'):
            loss = loss + model.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        for k in y:
            if k not in y_true_dict:
                y_true_dict[k] = []
                y_scores_dict[k] = []
            y_true_dict[k].append(batch[k][batch[k] != -1])
            y_scores_dict[k].append(y[k][batch[k] != -1].detach().cpu())
    
    logs = [f'Train loss: {np.mean(losses):.6f}']
    for k in y_true_dict:
        y_true = torch.cat(y_true_dict[k], dim = 0).detach().cpu()
        y_scores = torch.cat(y_scores_dict[k], dim = 0).detach().cpu()
        if k != 'age':
            y_true = y_true.numpy()
            y_scores = y_scores.numpy().argmax(1)
            acc = accuracy_score(y_true, y_scores)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
            logs.append(f'{k}-Accuracy: {acc:.6f}')
        else:
            logs.append(f'{k}-MSE: {torch.nn.functional.mse_loss(y_scores, y_true):.6f}')
    
    print(', '.join(logs))

def eval(model, experts, device, loader, args):
    model.eval()
    # y_true = []
    # y_scores = []
    y_true_dict = {}#{k: [] for k in LOSS_FUNCS}
    y_scores_dict = {}#{k: [] for k in LOSS_FUNCS}

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        with torch.no_grad():
            y = model(batch['x'].to(device), [embed.to(device) for embed in batch['expert_embed']], topk=args.topk_expert)
            # pred = classifier(feat, edge_index, batchid)
            # pred = pred['y']

        # if force_2class:
        #     _2cls_ind = torch.where(batch['y']<=1)[0]
        #     y['y'] = y['y'][_2cls_ind]
        #     batch['y'] = batch['y'][_2cls_ind]
    #     y_true.append(batch.y)
    #     y_scores.append(pred.detach().cpu())

    # y_true = torch.cat(y_true, dim = 0).detach().cpu().numpy()
    # y_scores = torch.cat(y_scores, dim = 0).numpy().argmax(1)
    # if hcpatoukb:
    #     y_scores[y_scores>1] = 1
    #     y_true[y_true>1] = 1
    # acc = accuracy_score(y_true, y_scores)
    # prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
    # return acc, prec, rec, f1

        for k in y:
            if k not in y_true_dict:
                y_true_dict[k] = []
                y_scores_dict[k] = []
            y_true_dict[k].append(batch[k][batch[k] != -1])
            y_scores_dict[k].append(y[k][batch[k] != -1].detach().cpu())

    scores = {}    
    for k in y_true_dict:
        y_true = torch.cat(y_true_dict[k], dim = 0).detach().cpu()
        y_scores = torch.cat(y_scores_dict[k], dim = 0).detach().cpu()
        if k != 'age':
            y_true = y_true.numpy()
            y_scores = y_scores.numpy()
            acc = accuracy_score(y_true, y_scores.argmax(1))
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores.argmax(1), average='weighted')
            fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1], pos_label=1)
            auc_score = auc(fpr, tpr)
            scores['y'] = [acc, prec, rec, f1, auc_score]
        else:
            r2 = torch.corrcoef(torch.stack([y_scores, y_true]))[0, 1].abs()
            scores[k] = [-1, -1, -1, torch.nn.functional.mse_loss(y_scores, y_true), r2]    

    return scores

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()