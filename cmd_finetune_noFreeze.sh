python finetune.py --dataname ppmi --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_ppmi_onlyAAL.log
python finetune.py --dataname taowu --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_taowu_onlyAAL.log
python finetune.py --dataname neurocon --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_neurocon_onlyAAL.log
python finetune.py --dataname adni --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_adni_onlyAAL.log

python finetune.py --dataname ppmi --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_ppmi_36expert.log
python finetune.py --dataname taowu --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_taowu_36expert.log
python finetune.py --dataname neurocon --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_neurocon_36expert.log
python finetune.py --dataname adni --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_adni_36expert.log

python finetune.py --dataname ppmi --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_ppmi_exAAL.log
python finetune.py --dataname taowu --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_taowu_exAAL.log
python finetune.py --dataname neurocon --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_neurocon_exAAL.log
python finetune.py --dataname adni --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-noFreeze_adni_exAAL.log

python finetune.py --use_pseudolabel --dataname ppmi --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_ppmi_onlyAAL.log
python finetune.py --use_pseudolabel --dataname taowu --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_taowu_onlyAAL.log
python finetune.py --use_pseudolabel --dataname neurocon --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_neurocon_onlyAAL.log
python finetune.py --use_pseudolabel --dataname adni --nofreeze_expert --device cuda:3 --expert_atlas AAL_116 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_adni_onlyAAL.log

python finetune.py --use_pseudolabel --dataname ppmi --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_ppmi_36expert.log
python finetune.py --use_pseudolabel --dataname taowu --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_taowu_36expert.log
python finetune.py --use_pseudolabel --dataname neurocon --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_neurocon_36expert.log
python finetune.py --use_pseudolabel --dataname adni --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_adni_36expert.log

python finetune.py --use_pseudolabel --dataname ppmi --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_ppmi_exAAL.log
python finetune.py --use_pseudolabel --dataname taowu --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_taowu_exAAL.log
python finetune.py --use_pseudolabel --dataname neurocon --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_neurocon_exAAL.log
python finetune.py --use_pseudolabel --dataname adni --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-bias-pseudoL-noFreeze_adni_exAAL.log


# python finetune.py --dataname abide --nofreeze_expert --device cuda:3 --expert_atlas Gordon_333 Shaefer_400 --model brainMoE > logs/ft_brainMoE-noFreeze_abide_exAAL.log
# python finetune.py --dataname abide --nofreeze_expert --device cuda:3 --model brainMoE > logs/ft_brainMoE-noFreeze_abide_36expert.log
