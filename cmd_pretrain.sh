python pretrain.py --tgt_atlas AAL_116 --device cuda:0 > pretrain_aal116_experts.log
python pretrain.py --tgt_atlas Gordon_333 --device cuda:2 --batch_size 128 > pretrain_gordon333_experts.log
python pretrain.py --tgt_atlas Shaefer_400 --device cuda:3 --batch_size 128 > pretrain_shaefer400_experts.log


python pretrain.py --model brainMoEDecoder --device cuda:4 > pretrain_aal116_experts_brainMoEDecoder.log
python pretrain.py --tgt_atlas Gordon_333 --model brainMoEDecoder --device cuda:0 --batch_size 128 > pretrain_gordon333_experts_brainMoEDecoder.log
python pretrain.py --tgt_atlas Shaefer_400 --model brainMoEDecoder --device cuda:2 --batch_size 128 > pretrain_shaefer400_experts_brainMoEDecoder.log