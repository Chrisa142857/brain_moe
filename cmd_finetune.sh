# python finetune.py --dataname ppmi --device cuda:4 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias_ppmi_onlyAAL.log
# python finetune.py --dataname taowu --device cuda:4 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias_taowu_onlyAAL.log
# python finetune.py --dataname neurocon --device cuda:4 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias_neurocon_onlyAAL.log
# python finetune.py --dataname adni --device cuda:4 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias_adni_onlyAAL.log

# python finetune.py --dataname ppmi --device cuda:4 --model brainMoE > logs/ft_brainMoE-bias_ppmi_36expert.log
# python finetune.py --dataname taowu --device cuda:4 --model brainMoE > logs/ft_brainMoE-bias_taowu_36expert.log
# python finetune.py --dataname neurocon --device cuda:4 --model brainMoE > logs/ft_brainMoE-bias_neurocon_36expert.log
# python finetune.py --dataname adni --device cuda:4 --model brainMoE > logs/ft_brainMoE-bias_adni_36expert.log

# python finetune.py --dataname ppmi --device cuda:4 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias_ppmi_exAAL.log
# python finetune.py --dataname taowu --device cuda:4 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias_taowu_exAAL.log
# python finetune.py --dataname neurocon --device cuda:4 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias_neurocon_exAAL.log
# python finetune.py --dataname adni --device cuda:4 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias_adni_exAAL.log


# python finetune.py --dataname ppmi --batch_size 32 --expert_atlas AAL_116 > finetune_V132L8H_ppmi_onlyAAL.log
# python finetune.py --dataname adni --batch_size 32 --expert_atlas AAL_116 --device cuda:2 > finetune_V132L8H_adni_onlyAAL.log
# python finetune.py --dataname taowu --batch_size 32 --expert_atlas AAL_116 --device cuda:2 > finetune_V132L8H_taowu_onlyAAL.log
# python finetune.py --dataname neurocon --batch_size 32 --expert_atlas AAL_116 --device cuda:2 > finetune_V132L8H_neurocon_onlyAAL.log
# python finetune.py --dataname abide --batch_size 32 --expert_atlas AAL_116 --device cuda:2 > finetune_V132L8H_abide_onlyAAL.log
# python finetune.py --dataname sz-diana --batch_size 32 --expert_atlas AAL_116 --device cuda:4 > finetune_V132L8H_sz-diana_onlyAAL.log

# python finetune.py --dataname ppmi --batch_size 32 --device cuda:3 > finetune_V132L8H_ppmi_36expert.log
# python finetune.py --dataname adni --batch_size 32 --device cuda:3 > finetune_V132L8H_adni_36expert.log
# python finetune.py --dataname taowu --batch_size 32 --device cuda:3 > finetune_V132L8H_taowu_36expert.log
# python finetune.py --dataname neurocon --batch_size 32 --device cuda:3 > finetune_V132L8H_neurocon_36expert.log
# python finetune.py --dataname abide --batch_size 32 --device cuda:3 > finetune_V132L8H_abide_36expert.log
# python finetune.py --dataname sz-diana --batch_size 32 --device cuda:3 > finetune_V132L8H_sz-diana_36expert.log


# python finetune.py --dataname ppmi  --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:4 > finetune_V132L8H_ppmi_exAAL.log
# python finetune.py --dataname adni  --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:4 > finetune_V132L8H_adni_exAAL.log
# python finetune.py --dataname taowu  --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:4 > finetune_V132L8H_taowu_exAAL.log
# python finetune.py --dataname neurocon  --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:4 > finetune_V132L8H_neurocon_exAAL.log
# python finetune.py --dataname abide  --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:4 > finetune_V132L8H_abide_exAAL.log
# python finetune.py --dataname sz-diana  --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:4 > finetune_V132L8H_sz-diana_exAAL.log



python finetune.py --dataname ppmi --model brainMoE --expert_atlas AAL_116 --batch_size 32 --device cuda:1 > finetune_V1-132L8H_ppmi_onlyAAL.log
python finetune.py --dataname adni --model brainMoE --expert_atlas AAL_116 --batch_size 32 --device cuda:1 > finetune_V1-132L8H_adni_onlyAAL.log
python finetune.py --dataname taowu --model brainMoE --expert_atlas AAL_116 --batch_size 32 --device cuda:1 > finetune_V1-132L8H_taowu_onlyAAL.log
python finetune.py --dataname neurocon --model brainMoE --expert_atlas AAL_116 --batch_size 32 --device cuda:1 > finetune_V1-132L8H_neurocon_onlyAAL.log
python finetune.py --dataname sz-diana --model brainMoE --expert_atlas AAL_116 --batch_size 32 --device cuda:1 > finetune_V1-132L8H_sz-diana_onlyAAL.log
python finetune.py --dataname abide --model brainMoE --expert_atlas AAL_116 --batch_size 32 --device cuda:1 > finetune_V1-132L8H_abide_onlyAAL.log
python finetune.py --dataname abide --expert_atlas AAL_116 --batch_size 20 --lr 0.00005 --device cuda:1 > finetune_V232L8H_abide_onlyAAL.log

python finetune.py --dataname ppmi --model brainMoE --batch_size 32 --device cuda:2 > finetune_V1-132L8H_ppmi_36expert.log
python finetune.py --dataname adni --model brainMoE --batch_size 32 --device cuda:2 > finetune_V1-132L8H_adni_36expert.log
python finetune.py --dataname taowu --model brainMoE --batch_size 32 --device cuda:2 > finetune_V1-132L8H_taowu_36expert.log
python finetune.py --dataname neurocon --model brainMoE --batch_size 32 --device cuda:2 > finetune_V1-132L8H_neurocon_36expert.log
python finetune.py --dataname sz-diana --model brainMoE --batch_size 32 --device cuda:2 > finetune_V1-132L8H_sz-diana_36expert.log
python finetune.py --dataname abide --model brainMoE --batch_size 32 --device cuda:2 > finetune_V1-132L8H_abide_36expert.log
python finetune.py --dataname abide --batch_size 20 --lr 0.00005 --device cuda:2 > finetune_V232L8H_abide_36expert.log


python finetune.py --dataname ppmi  --model brainMoE --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:3 > finetune_V1-132L8H_ppmi_exAAL.log
python finetune.py --dataname adni  --model brainMoE --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:3 > finetune_V1-132L8H_adni_exAAL.log
python finetune.py --dataname taowu  --model brainMoE --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:3 > finetune_V1-132L8H_taowu_exAAL.log
python finetune.py --dataname neurocon  --model brainMoE --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:3 > finetune_V1-132L8H_neurocon_exAAL.log
python finetune.py --dataname sz-diana  --model brainMoE --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:3 > finetune_V1-132L8H_sz-diana_exAAL.log
python finetune.py --dataname abide  --model brainMoE --expert_atlas Gordon_333 Shaefer_400 --batch_size 32 --device cuda:3 > finetune_V1-132L8H_abide_exAAL.log
python finetune.py --dataname abide  --expert_atlas Gordon_333 Shaefer_400 --batch_size 20 --lr 0.00005 --device cuda:3 > finetune_V232L8H_abide_exAAL.log



python finetune.py --dataname ppmi --expert_atlas AAL_116 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_ppmi_onlyAAL.log
python finetune.py --dataname adni --expert_atlas AAL_116 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_adni_onlyAAL.log
python finetune.py --dataname taowu --expert_atlas AAL_116 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_taowu_onlyAAL.log
python finetune.py --dataname neurocon --expert_atlas AAL_116 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_neurocon_onlyAAL.log
python finetune.py --dataname sz-diana --expert_atlas AAL_116 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_sz-diana_onlyAAL.log
python finetune.py --dataname ppmi --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_ppmi_36expert.log
python finetune.py --dataname adni --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_adni_36expert.log
python finetune.py --dataname taowu --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_taowu_36expert.log
python finetune.py --dataname neurocon --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_neurocon_36expert.log
python finetune.py --dataname sz-diana --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_sz-diana_36expert.log
python finetune.py --dataname ppmi  --expert_atlas Gordon_333 Shaefer_400 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_ppmi_exAAL.log
python finetune.py --dataname adni  --expert_atlas Gordon_333 Shaefer_400 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_adni_exAAL.log
python finetune.py --dataname taowu  --expert_atlas Gordon_333 Shaefer_400 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_taowu_exAAL.log
python finetune.py --dataname neurocon  --expert_atlas Gordon_333 Shaefer_400 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_neurocon_exAAL.log
python finetune.py --dataname sz-diana  --expert_atlas Gordon_333 Shaefer_400 --batch_size 20 --lr 0.00005 --device cuda:0 > finetune_V232L8H_sz-diana_exAAL.log



python finetune.py --dataname ppmi --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_ppmi_1expert.log
python finetune.py --dataname adni --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_adni_1expert.log
python finetune.py --dataname taowu --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_taowu_1expert.log
python finetune.py --dataname neurocon --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_neurocon_1expert.log
python finetune.py --dataname sz-diana --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_sz-diana_1expert.log
python finetune.py --dataname abide --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_abide_1expert.log

python finetune.py --dataname ppmi --model brainMoEDecoder --batch_size 200 --device cuda:3 --single_expert > finetune_V232L8H_ppmi_1expert.log
python finetune.py --dataname adni --model brainMoEDecoder --batch_size 200 --device cuda:3 --single_expert > finetune_V232L8H_adni_1expert.log
python finetune.py --dataname taowu --model brainMoEDecoder --batch_size 200 --device cuda:3 --single_expert > finetune_V232L8H_taowu_1expert.log
python finetune.py --dataname neurocon --model brainMoEDecoder --batch_size 200 --device cuda:3 --single_expert > finetune_V232L8H_neurocon_1expert.log
python finetune.py --dataname sz-diana --model brainMoEDecoder --batch_size 200 --device cuda:3 --single_expert > finetune_V232L8H_sz-diana_1expert.log
python finetune.py --dataname abide --model brainMoEDecoder --batch_size 200 --device cuda:3 --single_expert > finetune_V232L8H_abide_1expert.log


python finetune.py --dataname hcpa --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_hcpa_1expert.log
python finetune.py --dataname hcpya --model brainMoE --batch_size 200 --device cuda:3 --single_expert > finetune_V1-132L8H_hcpya_1expert.log
python finetune.py --dataname hcpa --model brainMoEDecoder --batch_size 200 --device cuda:2 --single_expert > finetune_V232L8H_hcpa_1expert.log
python finetune.py --dataname hcpya --model brainMoEDecoder --batch_size 200 --device cuda:3 --single_expert > finetune_V232L8H_hcpya_1expert.log



python finetune.py --dataname hcpa --train_obj sex --model brainMoE --batch_size 200 --device cuda:4 --single_expert > finetune_V1-132L8H_hcpa-sex_1expert.log
python finetune.py --dataname hcpya --train_obj sex --model brainMoE --batch_size 200 --device cuda:2 --single_expert > finetune_V1-132L8H_hcpya-sex_1expert.log
python finetune.py --dataname hcpa --train_obj sex --model brainMoEDecoder --batch_size 200 --device cuda:0 --single_expert > finetune_V232L8H_hcpa-sex_1expert.log
python finetune.py --dataname hcpya --train_obj sex --model brainMoEDecoder --batch_size 200 --device cuda:4 --single_expert > finetune_V232L8H_hcpya-sex_1expert.log


python finetune.py --dataname ppmi --train_obj sex --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_ppmi-sex_1expert.log
python finetune.py --dataname taowu --train_obj sex --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_taowu-sex_1expert.log
python finetune.py --dataname adni --train_obj sex --model brainMoE --batch_size 200 --device cuda:1 --single_expert > finetune_V1-132L8H_adni-sex_1expert.log
python finetune.py --dataname neurocon --train_obj sex --model brainMoE --batch_size 200 --device cuda:4 --single_expert > finetune_V1-132L8H_neurocon-sex_1expert.log
python finetune.py --dataname sz-diana --train_obj sex --model brainMoE --batch_size 200 --device cuda:4 --single_expert > finetune_V1-132L8H_sz-diana-sex_1expert.log
python finetune.py --dataname abide --train_obj sex --model brainMoE --batch_size 200 --device cuda:4 --single_expert > finetune_V1-132L8H_abide-sex_1expert.log



python finetune.py --dataname adni --model brainMoE --batch_size 200 --device cuda:1 --expert_atlas AAL_116 > finetune_V1-232L8H_adni_onlyAAL.log



python finetune.py --dataname hcpa --train_obj y --model brainMoE --batch_size 200 --device cuda:2 --expert_atlas AAL_116 > finetune_V1-432L8H_hcpa_onlyAAL.log
python finetune.py --dataname hcpya --train_obj sex --model brainMoE --batch_size 200 --device cuda:2 --expert_atlas AAL_116 > finetune_V1-432L8H_hcpya-sex_onlyAAL.log
python finetune.py --dataname abide --train_obj sex --model brainMoE --batch_size 200 --device cuda:2 --expert_atlas AAL_116 > finetune_V1-432L8H_abide-sex_onlyAAL.log
python finetune.py --dataname adni --train_obj sex --model brainMoE --batch_size 200 --device cuda:1 --expert_atlas AAL_116 > finetune_V1-432L8H_adni-sex_onlyAAL.log

python finetune.py --dataname ppmi --train_obj sex --model brainMoE --batch_size 200 --device cuda:4 --expert_atlas AAL_116 > finetune_V1-432L8H_ppmi-sex_onlyAAL.log
python finetune.py --dataname sz-diana --train_obj y --model brainMoE --batch_size 200 --device cuda:4 --expert_atlas AAL_116 > finetune_V1-432L8H_sz-diana_onlyAAL.log
python finetune.py --dataname hcpa --train_obj sex --model brainMoE --batch_size 200 --device cuda:4 --expert_atlas AAL_116 > finetune_V1-432L8H_hcpa-sex_onlyAAL.log
python finetune.py --dataname hcpya --train_obj y --model brainMoE --batch_size 200 --device cuda:4 --expert_atlas AAL_116 > finetune_V1-432L8H_hcpya_onlyAAL.log




python finetune_3typeExpert.py --dataname ppmi --train_obj sex --batch_size 200 --device cuda:0 > finetune_3typeExpert_ppmi-sex_onlyAAL.log
python finetune_3typeExpert.py --dataname taowu --train_obj sex --batch_size 200 --device cuda:0 > finetune_3typeExpert_taowu-sex_onlyAAL.log
python finetune_3typeExpert.py --dataname adni --train_obj sex --batch_size 200 --device cuda:0 > finetune_3typeExpert_adni-sex_onlyAAL.log
python finetune_3typeExpert.py --dataname neurocon --train_obj sex --batch_size 200 --device cuda:0 > finetune_3typeExpert_neurocon-sex_onlyAAL.log
python finetune_3typeExpert.py --dataname sz-diana --train_obj sex --batch_size 200 --device cuda:0 > finetune_3typeExpert_sz-diana-sex_onlyAAL.log
python finetune_3typeExpert.py --dataname abide --train_obj sex --batch_size 200 --device cuda:0 > finetune_3typeExpert_abide-sex_onlyAAL.log

python finetune_3typeExpert.py --dataname hcpa --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_hcpa-sex_onlyAAL.log

python finetune_3typeExpert.py --dataname hcpya --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_hcpya-sex_onlyAAL.log



python finetune_3typeExpert.py --dataname ppmi --train_obj y --batch_size 200 --device cuda:3 > finetune_3typeExpert_ppmi_onlyAAL.log
python finetune_3typeExpert.py --dataname taowu --train_obj y --batch_size 200 --device cuda:3 > finetune_3typeExpert_taowu_onlyAAL.log
python finetune_3typeExpert.py --dataname adni --train_obj y --batch_size 200 --device cuda:3 > finetune_3typeExpert_adni_onlyAAL.log
python finetune_3typeExpert.py --dataname neurocon --train_obj y --batch_size 200 --device cuda:3 > finetune_3typeExpert_neurocon_onlyAAL.log
python finetune_3typeExpert.py --dataname sz-diana --train_obj y --batch_size 200 --device cuda:3 > finetune_3typeExpert_sz-diana_onlyAAL.log
python finetune_3typeExpert.py --dataname abide --train_obj y --batch_size 200 --device cuda:3 > finetune_3typeExpert_abide_onlyAAL.log

python finetune_3typeExpert.py --dataname hcpa --train_obj y --batch_size 200 --device cuda:4 > finetune_3typeExpert_hcpa_onlyAAL.log

python finetune_3typeExpert.py --dataname hcpya --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_hcpya_onlyAAL.log




python finetune_3typeExpert.py --topk_expert 18 --dataname ppmi --train_obj y --batch_size 200 --device cuda:2 > finetune_3typeExpert_ppmi_topk18.log
python finetune_3typeExpert.py --topk_expert 18 --dataname taowu --train_obj y --batch_size 200 --device cuda:2 > finetune_3typeExpert_taowu_topk18.log
python finetune_3typeExpert.py --topk_expert 18 --dataname neurocon --train_obj y --batch_size 200 --device cuda:2 > finetune_3typeExpert_neurocon_topk18.log
python finetune_3typeExpert.py --topk_expert 18 --dataname sz-diana --train_obj y --batch_size 200 --device cuda:2 > finetune_3typeExpert_sz-diana_topk18.log

python finetune_3typeExpert.py --topk_expert 18 --dataname ppmi --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_ppmi-sex_topk18.log
python finetune_3typeExpert.py --topk_expert 18 --dataname taowu --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_taowu-sex_topk18.log
python finetune_3typeExpert.py --topk_expert 18 --dataname adni --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_adni-sex_topk18.log
python finetune_3typeExpert.py --topk_expert 18 --dataname neurocon --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_neurocon-sex_topk18.log
python finetune_3typeExpert.py --topk_expert 18 --dataname sz-diana --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_sz-diana-sex_topk18.log

python finetune_3typeExpert.py --topk_expert 18 --dataname abide --train_obj y --batch_size 200 --device cuda:2 > finetune_3typeExpert_abide_topk18.log

python finetune_3typeExpert.py --topk_expert 18 --dataname hcpa --train_obj y --batch_size 200 --device cuda:2 > finetune_3typeExpert_hcpa_topk18.log

python finetune_3typeExpert.py --topk_expert 18 --dataname hcpya --train_obj y --batch_size 200 --device cuda:2 > finetune_3typeExpert_hcpya_topk18.log

python finetune_3typeExpert.py --topk_expert 18 --dataname abide --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_abide-sex_topk18.log

python finetune_3typeExpert.py --topk_expert 18 --dataname hcpa --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_hcpa-sex_topk18.log

python finetune_3typeExpert.py --topk_expert 18 --dataname hcpya --train_obj sex --batch_size 200 --device cuda:2 > finetune_3typeExpert_hcpya-sex_topk18.log




python finetune_3typeExpert.py --topk_expert 9 --dataname ppmi --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_ppmi_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname taowu --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_taowu_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname adni --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_adni_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname neurocon --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_neurocon_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname sz-diana --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_sz-diana_topk9.log

python finetune_3typeExpert.py --topk_expert 9 --dataname ppmi --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_ppmi-sex_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname taowu --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_taowu-sex_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname adni --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_adni-sex_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname neurocon --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_neurocon-sex_topk9.log
python finetune_3typeExpert.py --topk_expert 9 --dataname sz-diana --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_sz-diana-sex_topk9.log

python finetune_3typeExpert.py --topk_expert 9 --dataname abide --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_abide_topk9.log

python finetune_3typeExpert.py --topk_expert 9 --dataname hcpa --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_hcpa_topk9.log

python finetune_3typeExpert.py --topk_expert 9 --dataname hcpya --train_obj y --batch_size 200 --device cuda:5 > finetune_3typeExpert_hcpya_topk9.log

python finetune_3typeExpert.py --topk_expert 9 --dataname abide --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_abide-sex_topk9.log

python finetune_3typeExpert.py --topk_expert 9 --dataname hcpa --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_hcpa-sex_topk9.log

python finetune_3typeExpert.py --topk_expert 9 --dataname hcpya --train_obj sex --batch_size 200 --device cuda:5 > finetune_3typeExpert_hcpya-sex_topk9.log


python finetune_3typeExpert.py --topk_expert 1 --dataname ppmi --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_ppmi_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname taowu --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_taowu_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname adni --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_adni_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname neurocon --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_neurocon_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname sz-diana --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_sz-diana_topk1.log

python finetune_3typeExpert.py --topk_expert 1 --dataname ppmi --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_ppmi-sex_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname taowu --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_taowu-sex_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname adni --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_adni-sex_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname neurocon --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_neurocon-sex_topk1.log
python finetune_3typeExpert.py --topk_expert 1 --dataname sz-diana --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_sz-diana-sex_topk1.log

python finetune_3typeExpert.py --topk_expert 1 --dataname abide --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_abide_topk1.log

python finetune_3typeExpert.py --topk_expert 1 --dataname hcpa --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_hcpa_topk1.log

python finetune_3typeExpert.py --topk_expert 1 --dataname hcpya --train_obj y --batch_size 200 --device cuda:1 > finetune_3typeExpert_hcpya_topk1.log

python finetune_3typeExpert.py --topk_expert 1 --dataname abide --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_abide-sex_topk1.log

python finetune_3typeExpert.py --topk_expert 1 --dataname hcpa --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_hcpa-sex_topk1.log

python finetune_3typeExpert.py --topk_expert 1 --dataname hcpya --train_obj sex --batch_size 200 --device cuda:1 > finetune_3typeExpert_hcpya-sex_topk1.log
