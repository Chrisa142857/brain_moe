import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
from typing import Optional#, List
import numpy as np


class RouterMultiModel(nn.Module):

    def __init__(self, experts, expert_tags, input_sz=116, hid_dim=2048, device='cuda:4', nlayer=4, head_num=4, nclass=2, use_topk=False, topk=5, use_pseudolabel=False, freeze_expert=True, target='y', activation="relu", dropout=0.1, normalize_before=False, return_intermediate=False):
        super().__init__()    
        nexpert = len(experts)
        ## V1-0, V1-1 ##########################
        # self.router = nn.Linear(input_sz**2, nexpert)
        ## V1-2 ################################ 79.3
        # self.router = nn.Sequential(
        #     nn.Linear(input_sz**2, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, nexpert)
        # )
        ## V1-3 ################################ not good
        # self.router = BrainExpertDecoder(
        #         input_sz=input_sz, 
        #         nclass=nexpert, 
        #         hid_dim=hid_dim, nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        ## V1-4 ################################ 79.7
        self.router = nn.Sequential(
            nn.Linear(input_sz**2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, nexpert)
        )
        ## V1-5 ################################ 78.4
        # self.router = nn.Sequential(
        #     nn.Linear(input_sz**2, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, nexpert)
        # )
        ## V1-8 ################################ not good
        # self.router = BrainExpertDecoder(
        #         input_sz=input_sz, 
        #         nclass=nexpert, 
        #         hid_dim=hid_dim, nlayer=4, head_num=8, activation=activation, dropout=dropout, normalize_before=normalize_before)
        print(self.router)
        ################################
        self.version = 'V1'
        self.target = target
        self.nclass = nclass
        self.freeze_expert = freeze_expert

        self.expert_tags = expert_tags
        self.nexpert = nexpert
        ## Expert adapter
        self.input_adapter_linear = nn.ModuleList([
                nn.Linear(expert_net.hid_dim, hid_dim)# if input_sz != expert_net.input_sz else nn.Identity() # V1-6: Indentity()
            for expert_net in experts
        ])

        
        ## V1.0 ###################
        # self.predicter = nn.ModuleList([
        #     BrainExpertDecoder(
        #     input_sz=input_sz, 
        #     nclass=nexpert if version == 'V1' else experts[0].nclass,
        #     # hid_dim=hid_dim, 
        #     hid_dim=experts[0].hid_dim*experts[0].nclass if version == 'V1' else experts[0].hid_dim,
        #     nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        # for _ in range(nclass)])
        
        ## V1.1~8 ###################
        predictor_ntoken = nexpert if self.version == 'V1' else experts[0].nclass*nexpert
        self.predicter = BrainExpertDecoder(
            input_sz=input_sz, 
            nclass=predictor_ntoken + nclass,
            # hid_dim=hid_dim, 
            hid_dim=hid_dim,
            nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        self.pred_query = nn.Embedding(nclass, hid_dim) # Nclass x C

        ## V1.9 ###################
        # self.predicter = nn.ModuleList([BrainExpertDecoder(
        #     input_sz=input_sz, 
        #     nclass=1 + nclass,
        #     # hid_dim=hid_dim, 
        #     hid_dim=experts[0].hid_dim,
        #     nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        # for i in range(nexpert)])
        # self.pred_query = nn.ModuleList([nn.Embedding(nclass, experts[0].hid_dim) for i in range(nexpert)]) # Nclass x C


        self.topk = topk
        self.use_pseudolabel = use_pseudolabel
        self.topk_batch_index = None

    def forward(self, x, org_expert_embeds, return_cross_attn=False, topk=-1):
        if self.topk_batch_index is None and topk != -1: 
            self.topk_batch_index = torch.arange(len(x))[:, None].repeat(1, topk).to(x.device)
        if self.topk_batch_index is not None:
            topk_batch_index = self.topk_batch_index[:len(x)]
        # x = data['x']
        ##### Linear ##########
        if not isinstance(self.router, BrainExpertDecoder):
            router_x = x.reshape(len(x), -1)
        else:
            router_x = x
        router_logits = self.router(router_x)
        if isinstance(router_logits, dict):
            router_logits = router_logits['y']
        router_logits = torch.softmax(router_logits, -1)
        # logits = self.fuse_experts(x, experts, router_logits)
        if topk != -1:
            topk_expert_index = (-1*router_logits).argsort(-1)[:, :topk]

        expert_embeds = []
        for experti in range(len(org_expert_embeds)):
            expert_embed = org_expert_embeds[experti]
            # expert_x = self.input_adapter_linear[experti](x)
            # if 'Mass' not in expert_tags[experti] and 'JEPA' not in expert_tags[experti]:
            #     expert_x = x
            # else:
            #     expert_x = data[expert_tags[experti].split('_')[0]]
            # if self.freeze_expert:
            #     with torch.no_grad():
            #         expert_embed = experts[experti](expert_x)['hidden_state'] # B x Ntoken x C (V1: Ntoken=1; V2: Ntoken=12)
            # else: # V1-7
            #     expert_embed = experts[experti](expert_x)['hidden_state'] # B x Ntoken x C (V1: Ntoken=1; V2: Ntoken=12)

            # expert_embed = self.cognition_adapter_linear(expert_embed.reshape(len(expert_embed), -1))
            
            if self.version == 'V1':
                expert_embed = expert_embed.reshape(len(expert_embed), -1)
            expert_embed = self.input_adapter_linear[experti](expert_embed)
            expert_embeds.append(expert_embed) # B x (Ntoken) x C
        # expert_embeds = self.cognition_adapter_linear(expert_embeds)
        if self.version == 'V1':
            expert_embeds = torch.stack(expert_embeds, 1) #* router_logits[..., None]
            
            if self.topk_batch_index is not None:
                router_logits = router_logits[topk_batch_index, topk_expert_index]
                expert_embeds = expert_embeds[topk_batch_index, topk_expert_index]
            ## V1.0 ###################
            # logits = []
            # for i in range(self.nclass):
            #     logit = self.predicter[i](x, query_embed=expert_embeds)['y'] # B x Nexpert
            #     logit = logit * router_logits 
            #     logits.append(logit.max(1)[0]) # B
            # logits = torch.stack(logits, -1) # B x Nclass
            ## V1.1~8 ###################
            query_embed = torch.cat([
                self.pred_query.weight.unsqueeze(0).repeat(len(x), 1, 1),  # B x Nclass x C
                expert_embeds * router_logits[..., None]], # B x Nexpert x C
                1)
            logits = self.predicter(x, query_embed=query_embed)['y'][:, :self.nclass] # B x Nclass
            ## V1.9 ###################
            # logits = []
            # for experti in range(expert_embeds.shape[1]):
            #     query_embed = torch.cat([
            #         self.pred_query[experti].weight.unsqueeze(0).repeat(len(x), 1, 1),  # B x Nclass x C
            #         expert_embeds[:, experti:experti+1]], # B x 1 x C
            #         1)
            #     logits.append(self.predicter[experti](x, query_embed=query_embed)['y'][:, :self.nclass] * router_logits[:, experti, None]) # B x Nclass
            # logits = torch.stack(logits).mean(0)

        if not return_cross_attn:
            return {self.target: logits}
        # else:
        #     return {'y': logits, 'attn': router_out['attn']}


class Router(nn.Module):

    def __init__(self, experts, expert_tags, input_sz=116, hid_dim=2048, device='cuda:4', nlayer=4, head_num=4, nclass=2, use_topk=False, topk=5, use_pseudolabel=False, freeze_expert=True, target='y', activation="relu", dropout=0.1, normalize_before=False, return_intermediate=False):
        super().__init__()    
        nexpert = len(experts)
        ## V1-0, V1-1 ##########################
        # self.router = nn.Linear(input_sz**2, nexpert)
        ## V1-2 ################################ 79.3
        # self.router = nn.Sequential(
        #     nn.Linear(input_sz**2, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, nexpert)
        # )
        ## V1-3 ################################ not good
        # self.router = BrainExpertDecoder(
        #         input_sz=input_sz, 
        #         nclass=nexpert, 
        #         hid_dim=hid_dim, nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        ## V1-4 ################################ 79.7
        self.router = nn.Sequential(
            nn.Linear(input_sz**2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, nexpert)
        )
        ## V1-5 ################################ 78.4
        # self.router = nn.Sequential(
        #     nn.Linear(input_sz**2, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, nexpert)
        # )
        ## V1-8 ################################ not good
        # self.router = BrainExpertDecoder(
        #         input_sz=input_sz, 
        #         nclass=nexpert, 
        #         hid_dim=hid_dim, nlayer=4, head_num=8, activation=activation, dropout=dropout, normalize_before=normalize_before)
        print(self.router)
        ################################
        if isinstance(experts[0], BrainExpert):
            version = 'V1'
        elif isinstance(experts[0], BrainExpertDecoder):
            version = 'V2'

        self.version = version
        self.target = target
        self.nclass = nclass
        self.freeze_expert = freeze_expert

        self.expert_tags = expert_tags
        self.nexpert = nexpert
        ## Expert adapter
        # self.expert_nclass = []
        # expert_adapter = []
        # for expert_net in experts:
        #     adapter = copy.deepcopy(expert_net)
        #     for p in adapter.parameters():
        #         p.requires_grad = not freeze_expert

        #     self.expert_nclass.append(len(adapter.object_query.weight))
        #     adapter.object_query.weight = torch.nn.Parameter(torch.cat([adapter.object_query.weight, torch.nn.Parameter(torch.randn(nclass, hid_dim), requires_grad=True).to(expert_net.object_query.weight.device)]))
        #     expert_adapter.append(adapter)
        # self.expert_adapter = nn.ModuleList(expert_adapter)
        self.input_adapter_linear = nn.ModuleList([
                nn.Linear(input_sz, expert_net.input_sz)# if input_sz != expert_net.input_sz else nn.Identity() # V1-6: Indentity()
            for expert_net in experts
        ])
        # self.input_adapter_linear = nn.ModuleList([
        #     BrainExpertDecoder(
        #         input_sz=input_sz, 
        #         nclass=expert_net.input_sz, 
        #         hid_dim=hid_dim, nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        #     for expert_net in experts
        # ])
        # cog_bin_num = 100
        # self.cognition_bins = torch.linspace(0, 1, cog_bin_num).to(device)
        # self.cognition_embed = nn.Embedding(len(self.cognition_bins)+1, hid_dim)
        # self.cognition_embed = nn.Embedding(12, hid_dim)
        # self.cognition_embed = nn.ModuleList([
        #     nn.Embedding(expert_net.nclass, hid_dim)
        #     for expert_net in experts
        # ])
        # self.input_adapter_linear = nn.Sequential(
        #         nn.Linear(input_sz, hid_dim),
        #         # nn.ReLU(),
        #         # nn.Linear(hid_dim, expert_net.input_sz)
        #     )
        
        # self.cognition_adapter_linear = nn.ModuleList([nn.Sequential(
        #         nn.Linear(expert_net.hid_dim*expert_net.nclass, hid_dim),
        #         # nn.ReLU(),
        #         # nn.Linear(hid_dim, hid_dim),
        #     )
        #     for expert_net in experts
        # ])
        # self.cognition_adapter_linear = nn.Linear(experts[0].hid_dim*experts[0].nclass, hid_dim)
        self.use_topk = use_topk
        
        ## V1.0 ###################
        # self.predicter = nn.ModuleList([
        #     BrainExpertDecoder(
        #     input_sz=input_sz, 
        #     nclass=nexpert if version == 'V1' else experts[0].nclass,
        #     # hid_dim=hid_dim, 
        #     hid_dim=experts[0].hid_dim*experts[0].nclass if version == 'V1' else experts[0].hid_dim,
        #     nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        # for _ in range(nclass)])
        
        ## V1.1~8 ###################
        predictor_ntoken = nexpert if version == 'V1' else experts[0].nclass*nexpert
        self.predicter = BrainExpertDecoder(
            input_sz=input_sz, 
            nclass=predictor_ntoken + nclass,
            # hid_dim=hid_dim, 
            hid_dim=experts[0].hid_dim,
            nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        self.pred_query = nn.Embedding(nclass, experts[0].hid_dim) # Nclass x C

        ## V1.9 ###################
        # self.predicter = nn.ModuleList([BrainExpertDecoder(
        #     input_sz=input_sz, 
        #     nclass=1 + nclass,
        #     # hid_dim=hid_dim, 
        #     hid_dim=experts[0].hid_dim,
        #     nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)
        # for i in range(nexpert)])
        # self.pred_query = nn.ModuleList([nn.Embedding(nclass, experts[0].hid_dim) for i in range(nexpert)]) # Nclass x C

        # self.predicter = BrainExpertDecoder(
        #     input_sz=hid_dim, 
        #     nclass=nclass, 
        #     hid_dim=hid_dim, nlayer=8, head_num=8, activation=activation, dropout=dropout, normalize_before=normalize_before)
        
        # self.predicter = BrainExpertDecoder(
        #     input_sz=input_sz, 
        #     nclass=nexpert, token_out_sz=nclass,
        #     hid_dim=hid_dim, nlayer=32, head_num=8, activation=activation, dropout=dropout, normalize_before=normalize_before)
        

        # self.pred_adapter_linear = nn.ModuleList([
        #     BrainExpertDecoder(
        #         input_sz=hid_dim, 
        #         nclass=nclass, 
        #         hid_dim=hid_dim, nlayer=nlayer, head_num=head_num, activation=activation, dropout=dropout, normalize_before=normalize_before)for expert_net in experts
        # ])

        self.topk = topk
        self.use_pseudolabel = use_pseudolabel

    def forward(self, x, experts, return_cross_attn=False):
        ##### CA ##########
        # router_out = self.router(x, return_cross_attn=return_cross_attn)
        # router_logits = router_out['y']
        ##### Linear ##########
        if not isinstance(self.router, BrainExpertDecoder):
            router_x = x.reshape(len(x), -1)
        else:
            router_x = x
        router_logits = self.router(router_x)
        if isinstance(router_logits, dict):
            router_logits = router_logits['y']
        router_logits = torch.softmax(router_logits, -1)
        # logits = self.fuse_experts(x, experts, router_logits)

        expert_embeds = []
        for experti in range(len(experts)):
            expert_x = self.input_adapter_linear[experti](x)
            if self.freeze_expert:
                with torch.no_grad():
                    expert_embed = experts[experti](expert_x)['hidden_state'] # B x Ntoken x C (V1: Ntoken=1; V2: Ntoken=12)
            else: # V1-7
                expert_embed = experts[experti](expert_x)['hidden_state'] # B x Ntoken x C (V1: Ntoken=1; V2: Ntoken=12)

            # expert_embed = self.cognition_adapter_linear(expert_embed.reshape(len(expert_embed), -1))
            
            if self.version == 'V1':
                expert_embed = expert_embed.reshape(len(expert_embed), -1)
            expert_embeds.append(expert_embed) # B x (Ntoken) x C
        # expert_embeds = self.cognition_adapter_linear(expert_embeds)
        if self.version == 'V1':
            expert_embeds = torch.stack(expert_embeds, 1) #* router_logits[..., None]
            ## V1.0 ###################
            # logits = []
            # for i in range(self.nclass):
            #     logit = self.predicter[i](x, query_embed=expert_embeds)['y'] # B x Nexpert
            #     logit = logit * router_logits 
            #     logits.append(logit.max(1)[0]) # B
            # logits = torch.stack(logits, -1) # B x Nclass
            ## V1.1~8 ###################
            query_embed = torch.cat([
                self.pred_query.weight.unsqueeze(0).repeat(len(x), 1, 1),  # B x Nclass x C
                expert_embeds * router_logits[..., None]], # B x Nexpert x C
                1)
            logits = self.predicter(x, query_embed=query_embed)['y'][:, :self.nclass] # B x Nclass
            ## V1.9 ###################
            # logits = []
            # for experti in range(expert_embeds.shape[1]):
            #     query_embed = torch.cat([
            #         self.pred_query[experti].weight.unsqueeze(0).repeat(len(x), 1, 1),  # B x Nclass x C
            #         expert_embeds[:, experti:experti+1]], # B x 1 x C
            #         1)
            #     logits.append(self.predicter[experti](x, query_embed=query_embed)['y'][:, :self.nclass] * router_logits[:, experti, None]) # B x Nclass
            # logits = torch.stack(logits).mean(0)

        elif self.version == 'V2':
            query_embed = torch.cat([
                self.pred_query.weight.unsqueeze(0).repeat(len(x), 1, 1),  # B x Nclass x C
                ] + [expert_embed * router_logits[:, experti][:, None, None] for experti, expert_embed in enumerate(expert_embeds)], # B x Ntoken x C
                1)
            logits = self.predicter(x, query_embed=query_embed)['y'][:, :self.nclass] # B x Nclass
        # logits = self.predicter(x, query_embed=expert_embeds)['y']
        # logits = logits * router_logits[..., None]
        # logits = logits.mean(1)

        if not return_cross_attn:
            return {self.target: logits}
        # else:
        #     return {'y': logits, 'attn': router_out['attn']}

    def fuse_experts(self, x, experts, router_logits):
        B = len(x)
        if self.use_pseudolabel: 
            self.loss = 0
            loss_group = [[] for _ in range(max(self.expert_group_ind)+1)]
        logits = []
        # x = self.input_adapter_linear(x)
        # cognition_embeddings = []
        # expert_outs = []
        # cognitions = []
        for experti in range(len(experts)):
            expert_x = self.input_adapter_linear[experti](x)
            # x = self.input_adapter_linear[experti](x, return_cross_attn=True)['attn'][:, -1]
            expert_logits = self.expert_adapter[experti](expert_x)['y']
            if self.use_pseudolabel:
                cognition_logits_per_expert = expert_logits[:, :self.expert_nclass[experti]]
                loss_group[self.expert_group_ind[experti]].append(cognition_logits_per_expert)
            logits_per_expert = expert_logits[:, self.expert_nclass[experti]:]
            logits.append(logits_per_expert * router_logits[:, experti][:, None])
            # x = self.input_adapter_linear[experti](x)
            # with torch.no_grad():
            #     expert_out = experts[experti](x, return_cross_attn=True)
            #     expert_outs.append(expert_out)
                # cognition.append(expert_out['y'])
        # cognition = torch.cat(cognition, -1).argmax(-1)
        # self.cognition_embed(cognition)
        # x = self.input_adapter_linear(x)
        # for expert_out in expert_outs:
            # cognition = expert_out['attn'][:, -1].reshape(B, -1) # B x N*P
            # cognition_embedding = self.cognition_embed(torch.bucketize(cognition, self.cognition_bins))
            # cognition = expert_out['y'] # B x expert_Nclass
            # cognition_embedding = self.cognition_embed[experti](cognition)
            # cognition_embedding = self.cognition_adapter_linear[experti](cognition) # B x C
            # cognition_embeddings.append(cognition_embedding)
            # logits_per_expert = self.pred_adapter_linear(x, query_embed=cognition_embedding)['y'] # B x Nclass
            # logits_per_expert = self.pred_adapter_linear[experti](x, query_embed=cognition_embedding)['y'] # B x Nclass
            # logits.append(logits_per_expert * router_logits[:, experti][:, None])
            # cognitions.append(expert_out['y'] * router_logits[:, experti][:, None])
            # cognitions.append(cognition)
        # cognition_embeddings = torch.stack(cognition_embeddings, 1) # B x Ncog x C
        
        if self.use_pseudolabel: 
            for gi in range(len(loss_group)):
                semisupervise = torch.cat(loss_group[gi], 1)
                self.loss += self.loss_fn(semisupervise, self.pseudolabels[gi].repeat(len(semisupervise)))
        logits = torch.stack(logits, 1) # B x Nexpert x Nclass
        if self.use_topk:
            # cognition_embeddings = torch.gather(cognition_embeddings, 1, (router_logits*-1).argsort(-1))[:, :, :self.topk]
            logits = torch.gather(logits, 1, (router_logits*-1).argsort(-1))[:, :, :self.topk]
        logits = logits.mean(1) # B x Nclass
        # cognitions = torch.stack(cognitions, 1) # B x Ncog x N*P
        # if cognitions.shape[-1] == 1:
        #     cognitions = cognitions.squeeze(-1).argmax(-1)#.repeat(self.nexpert, 1).T
        # else:
        #     cognitions = cognitions.argmax(-1)
        # cognitions_embed = self.cognition_embed(cognitions)
        # logits = self.pred_adapter_linear(cognition_embeddings)['y']
        # logits = self.pred_adapter_linear(self.input_adapter_linear(x) + cognitions_embed[:, None])['y']
        # logits = self.pred_adapter_linear(self.input_adapter_linear(x))['y']
        return logits


class BrainMoEDecoder(nn.Module):
    def __init__(self, expert_tag=[], **kwargs):
        super().__init__()
        # expert_n = len(expert_tag)
        self.experts = nn.ModuleList([
            BrainExpertDecoder(input_sz=int(t.split('_')[-1]), **kwargs)
            for t in expert_tag
        ])
        self.expert_tag = expert_tag
    
    def forward(self, x, expert_tag, return_cross_attn=False):
        outputs = []
        for t in expert_tag:
            out = self.experts[self.expert_tag.index(t)](x, return_cross_attn=return_cross_attn)
            out['expert_tag'] = t
            outputs.append(out)
        return outputs

class BrainExpertDecoder(nn.Module):

    def __init__(self, input_sz=116, hid_dim=2048, nlayer=4, head_num=4, nclass=12, token_out_sz=1, activation="relu", dropout=0.1, normalize_before=False, return_intermediate=False):
        super().__init__()    
        self.hid_dim = hid_dim
        self.input_sz = input_sz
        self.nclass = nclass
        self.input_linear = nn.Linear(input_sz, hid_dim)
        self.object_query = nn.Embedding(nclass, hid_dim)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayerWithSA(hid_dim, head_num, hid_dim,
                                                dropout, activation, normalize_before),
            nlayer,
            nn.LayerNorm(hid_dim), # None#
            return_intermediate=return_intermediate
        )
        self.pred_embed = nn.Linear(hid_dim, token_out_sz)


    def forward(self, x, mask=None, return_cross_attn=False, query_embed=None):
        x = self.input_linear(x)
        if query_embed is None:
            query_embed = self.object_query.weight
            query_embed = query_embed.unsqueeze(0).repeat(len(x), 1, 1)

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, x, memory_key_padding_mask=mask, query_pos=query_embed, return_cross_attn=return_cross_attn) # B X N X C
        if return_cross_attn:
            hs, attn = hs

        logits = self.pred_embed(hs).squeeze(-1)
        if not return_cross_attn:
            return {'y': logits, 'hidden_state': hs}
        else:
            return {'y': logits, 'hidden_state': hs, 'attn': attn}
        
class BrainMoE(nn.Module):
    def __init__(self, expert_tag=[], **kwargs):
        super().__init__()
        # expert_n = len(expert_tag)
        self.experts = nn.ModuleList([
            BrainExpert(input_sz=int(t.split('_')[-1]), **kwargs)
            for t in expert_tag
        ])
        self.expert_tag = expert_tag
    
    def forward(self, x, expert_tag, return_cross_attn=False):
        outputs = []
        for t in expert_tag:
            out = self.experts[self.expert_tag.index(t)](x, return_cross_attn=return_cross_attn)
            out['expert_tag'] = t
            outputs.append(out)
        return outputs

class BrainExpert(nn.Module):

    def __init__(self, input_sz=116, hid_dim=2048, nlayer=4, head_num=4, activation="relu", dropout=0.1, normalize_before=False, return_intermediate=False):
        super().__init__()    
        self.hid_dim = hid_dim
        self.input_sz = input_sz
        self.nclass = 1
        self.input_linear = nn.Linear(input_sz, hid_dim)
        self.object_query = nn.Embedding(1, hid_dim)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(hid_dim, head_num, hid_dim,
                                                dropout, activation, normalize_before),
            nlayer,
            nn.LayerNorm(hid_dim), # None#
            return_intermediate=return_intermediate
        )
        self.pred_embed = nn.Linear(hid_dim, 1)


    def forward(self, x, mask=None, return_cross_attn=False):
        x = self.input_linear(x)
        query_embed = self.object_query.weight
        query_embed = query_embed.unsqueeze(0).repeat(len(x), 1, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, x, memory_key_padding_mask=mask, query_pos=query_embed, return_cross_attn=return_cross_attn) # B X N X C
        if return_cross_attn:
            hs, attn = hs

        logits = self.pred_embed(hs).squeeze(-1)#.squeeze(-1)#.sigmoid()
        if not return_cross_attn:
            return {'y': logits, 'hidden_state': hs}
        else:
            return {'y': logits, 'hidden_state': hs, 'attn': attn}

        
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_cross_attn=False):
        output = tgt

        intermediate = []
        attn = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           return_cross_attn=return_cross_attn)
            if return_cross_attn:
                attn.append(output[1])
                output = output[0]
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            
            if not return_cross_attn:
                return torch.stack(intermediate)
            else:
                return torch.stack(intermediate), torch.stack(attn, dim=1) # B x L x P x N
        if not return_cross_attn:
            return output#.unsqueeze(0)
        else:
            return output, torch.stack(attn, dim=1) # B x L x P x N

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_cross_attn=False):
        # q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = tgt
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if not return_cross_attn:
            return tgt
        else:
            return tgt, cross_attn

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_cross_attn=False):
        tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if not return_cross_attn:
            return tgt
        else:
            return tgt, cross_attn

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_cross_attn=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_cross_attn)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_cross_attn)


class TransformerDecoderLayerWithSA(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_cross_attn=False):
        q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if not return_cross_attn:
            return tgt
        else:
            return tgt, cross_attn

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_cross_attn=False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if not return_cross_attn:
            return tgt
        else:
            return tgt, cross_attn

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_cross_attn=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_cross_attn)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_cross_attn)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
