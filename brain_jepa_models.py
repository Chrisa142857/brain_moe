# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

from logging import getLogger

logger = getLogger()


class VisionTransformerEval(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, model_name='vit_base', attn_mode='flash_attn', global_pool=False, add_w=False, device=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=None, num_classes=2):
        super(VisionTransformerEval, self).__init__()
        
        gradient_csv_path = '/ram/USERS/ziquanw/Brain-JEPA/data/gradient_mapping_116_moe.csv'#args.gradient_path#'data/gradient_mapping_450.csv'
        def load_gradient():
            df = pd.read_csv(gradient_csv_path, header=None)
            gradient = torch.tensor(df.values, dtype=torch.float32)
            return gradient.unsqueeze(0)

        gradient = load_gradient().to(device, non_blocking=True)
        
        # from src.helper import init_model
        self.encoder, _ = init_model(
            device=device,
            patch_size=16, #args.patch_size, # 49
            crop_size=[116,160],#args.crop_size, # (450, 490)
            pred_depth=12, #args.pred_depth, # 12
            pred_emb_dim=384, #args.pred_emb_dim, # 384
            model_name=model_name,
            gradient_pos_embed=gradient,
            attn_mode=attn_mode,
            add_w=add_w,
            gradient_checkpointing=True)
        
        self.gradient_checkpointing = True

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(self.encoder.embed_dim)        
        
        self.head = nn.Linear(self.encoder.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.hid_dim = 768

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):

        x = self.encoder(x)
        if self.global_pool:
            x = x[:, :, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            outcome = x[:, 0]

        # if self.gradient_checkpointing:
        #     try:
        #         x = torch.utils.checkpoint.checkpoint(self.head, outcome, use_reentrant=False)
        #     except ValueError as e:
        #         print(1)
        # else:
        #     x = self.head(outcome)

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=1, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, in_chans=1, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
    gradient_pos_embed=None,
    attn_mode='normal',
    add_w=False,
    gradient_checkpointing=False,
    use_mae=False
):
    encoder = vit_base(
        img_size=(crop_size[0], crop_size[1]),
        patch_size=patch_size,
        in_chans=1,
        gradient_pos_embed=gradient_pos_embed,
        attn_mode=attn_mode,
        add_w=add_w,
        embed_dim=768 if not use_mae else crop_size[1],
        gradient_checkpointing=gradient_checkpointing)
    predictor = vit_predictor(
        num_patches=encoder.patch_embed.num_patches,
        num_patches_2d=encoder.patch_embed.num_patches_2d,
        embed_dim=encoder.embed_dim if not use_mae else crop_size[1],
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        gradient_pos_embed=gradient_pos_embed,
        attn_mode=attn_mode,
        add_w=add_w)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)
    return encoder, predictor



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


class GradTs_2dPE(nn.Module):
    def __init__(self, in_chan, embed_dim, grid_size, add_w=False, cls_token=False) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.grid = self.get_grid(grid_size)
        self.emb_h = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1], embed_dim // 2), requires_grad=False)
        pos_emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, self.grid[0])  # (H*W, D/2)
        self.emb_h.data.copy_(torch.from_numpy(pos_emb_h).float())
        
        self.add_w = add_w
        if add_w == 'origin':
            self.emb_w = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1], embed_dim // 2), requires_grad=False)
            pos_emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, self.grid[1])  # (H*W, D/2)
            self.emb_w.data.copy_(torch.from_numpy(pos_emb_w).float())
            
        if add_w == 'mapping':
            self.predictor_pos_embed_proj = nn.Linear(in_chan, embed_dim//2)
        self.cls_token = cls_token

        
    def get_grid(self, grid_size):    
        grid_h = np.arange(grid_size[0], dtype=float)
        grid_w = np.arange(grid_size[1], dtype=float)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
        return grid
    
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega   # (D/2,)

        pos = pos.reshape(-1)   # (M,)
        out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb
    
    def forward(self, gradient):  # TODO: check which one is emb_w and which one is emb_h !!!!!
        if self.add_w == 'mapping':
            gradient_pos_embed = self.predictor_pos_embed_proj(gradient)
            # emb_w = torch.cat([gradient_pos_embed.squeeze()]*10, dim=0)  # (H*W, D/2)
            # print(self.emb_h.shape)
            emb_w = gradient_pos_embed.squeeze().repeat_interleave(10, dim=0)
            emb_w = (emb_w - emb_w.min()) / (emb_w.max() - emb_w.min()) * 2 - 1
            
        if self.add_w == 'mapping':
            emb_w = emb_w
        elif self.add_w == 'origin':
            emb_w = self.emb_w
        else:
            raise Exception('self.add_w error')
        
            
        emb = torch.cat([self.emb_h, emb_w], dim=1).unsqueeze(0)  # (H*W, D)
        
        if self.cls_token:
            pos_embed = torch.concat([torch.zeros([1, 1, emb.shape[2]]).cuda(), emb], dim=1)
        else:
            pos_embed = emb
        return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_mode='normal'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_rate = proj_drop
        
        self.attn_mode = attn_mode

    def forward(self, x, return_attn=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        if self.attn_mode == 'normal':
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2)
        elif self.attn_mode == 'flash_attn':
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.proj_drop_rate)
            if return_attn:
                x, attn, _ = flash_attn_qkvpacked_func(qkv, dropout_p=self.proj_drop_rate)
        else:
            raise Exception('error')
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x, None


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_mode='normal'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_mode=attn_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x), return_attention)

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(450, 490), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0]) * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_2d = (img_size[0], img_size[1] // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        num_patches_2d,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        gradient_pos_embed=None,
        attn_mode='normal',
        add_w=False,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # -- predictor_pos_embedding gradient
        self.gradient_pos_embed = gradient_pos_embed
        self.predictor_2dpe_proj = GradTs_2dPE(gradient_pos_embed.shape[-1], predictor_embed_dim, num_patches_2d, add_w=add_w, cls_token=False)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attn_mode=attn_mode)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, return_attention=False):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        predictor_pos_embed = self.predictor_2dpe_proj(self.gradient_pos_embed)
        
        x_pos_embed = predictor_pos_embed.repeat(B, 1, 1)

        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        attn_set = []
        for blk in self.predictor_blocks:
            if return_attention:
                x, attn = blk(x, return_attention)
                attn_set.append(attn.detach().cpu())
            else:
                x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        if return_attention:
            return x, attn_set
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=(224,224),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        gradient_pos_embed=None,
        attn_mode='normal',
        add_w=False,
        gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_2d = self.patch_embed.num_patches_2d
        # -- gradient pos embedding
        self.gradient_pos_embed = gradient_pos_embed
        self.pos_embed_proj = GradTs_2dPE(gradient_pos_embed.shape[-1], embed_dim, self.num_patches_2d, add_w=add_w, cls_token=False)
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attn_mode=attn_mode)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None, return_attention=False):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.pos_embed_proj(self.gradient_pos_embed)
        # pos_embed = self.interpolate_pos_encoding(x, pos_embed)
        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        attn_set = []
        for i, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                if return_attention:
                    x, attn = torch.utils.checkpoint.checkpoint(blk, x, return_attention, use_reentrant=False)
                    attn_set.append(attn.detach().cpu())
                else:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                if return_attention:
                    x, attn = blk(x, return_attention)
                    attn_set.append(attn.detach().cpu())
                else:
                    x = blk(x)

        if self.norm is not None:
            x = self.norm(x)
        
        if return_attention:
            return x, attn_set
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
}

import random
class EvalTransformation:
    def __init__(self,crop_size=[450, 490],use_normalization=False, downsample=False,target='y',
        sampling_rate=3,
        num_frames=160,):
        self.size = crop_size
        self.target = target
        self.use_normalization = use_normalization
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        # self.seq_length = 490

    def __call__(self,data):
        img = data['x'] # N x T (one brain)
        # img = torch.randn(self.size)
        # img = torch.nn.functional.interpolate(img[None, None], size=self.size, mode='bilinear')[0, 0]
        # img = torch.cat([img, torch.zeros(self.size[0]-img.shape[0], img.shape[1])])
        input_x  = img[:, (img!=0).any(0)]
        seq_length = input_x.shape[1]

        label_y = data[self.target]#.item()
        

        if self.use_normalization:
            mean = input_x.mean()
            std = input_x.std()
            input_x = (input_x - mean) / std
        input_x[input_x.isnan()] = 0
        if self.downsample:
            clip_size = self.sampling_rate * self.num_frames
            # clip_size = self.num_frames
            start_idx, end_idx = self._get_start_end_idx(seq_length, clip_size)
            # print(start_idx, end_idx)
            ts_array = self._temporal_sampling(
                        input_x, start_idx, end_idx, self.num_frames
                    )
            input_x = torch.unsqueeze(ts_array, 0).to(torch.float32)
        else:   
            input_x = torch.unsqueeze(input_x, 0).to(torch.float32) 
        if self.target != 'age':
            return input_x.to(torch.float32), int(label_y)
        else:
            return input_x.to(torch.float32), torch.FloatTensor(label_y)/100
 
    def _get_start_end_idx(self, fmri_size, clip_size):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Sample a clip of size clip_size from a video of size video_size and
        return the indices of the first and last frame of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the video to
        num_clips clips, and select the start and end index of clip_idx-th video
        clip.
        Args:
            video_size (int): number of overall frames.
            clip_size (int): size of the clip to sample from the frames.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the video to num_clips
                clips, and select the start and end index of the clip_idx-th video
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given video for testing.
        Returns:
            start_idx (int): the start frame index.
            end_idx (int): the end frame index.
        """
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx
    
    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Given the start and end frame index, sample num_samples frames between
        the start and end with equal interval.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num video frames` x `channel` x `height` x `width`.
            start_idx (int): the index of the start frame.
            end_idx (int): the index of the end frame.
            num_samples (int): number of frames to sample.
        Returns:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `num clip frames` x `channel` x `height` x `width`.
        """
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames
    
