## Codes for Brain-Environment Cross-Attention (BECA).

The official implementation of paper "BrainMoE: Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model". Pre-trained model weights can be found here [https://drive.google.com/drive/folders/1Gc5SmsQMaNZHxi8W6LBnU7mYQXhbi_ok?usp=sharing](https://drive.google.com/drive/folders/1Gc5SmsQMaNZHxi8W6LBnU7mYQXhbi_ok?usp=sharing).

### File structure


```
.
├── brain_jepa_models.py        # BrainJEPA finetune model
├── brain_mass_models.py        # BrainMass finetune model
├── brainmoe_in_markdown.md     # Log in markdown tables
├── cmd_finetune.sh             # cmd for finetune
├── cmd_pretrain.sh             # cmd for pretrain
├── datasets.py                 # datasets
├── finetune_3typeExpert.py     # finetune all-in-one
├── finetune.py                 # finetune brainMoE
├── models.py                   # BrainMoE models
├── parse_logs.py               # convert log files
├── plot_topk_moe.py            # plot ablation studies
├── prepare_pretrain_data.py    # prepare pretrain data
├── pretrain.py                 # pretrain experts
└── README.md
```