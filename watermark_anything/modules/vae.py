# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# These classes and their supporting functions below are lightly adapted from the VQGAN repo available at: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py # noqa

from einops import rearrange
import numpy as np

import torch
import torch.nn as nn

from .common import MLPBlock


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class ConditionalNorm(nn.Module):
    """
    Vestigit 风格的条件归一化层。
    使用水印向量 (message_emb) 预测缩放 (gamma) 和偏移 (beta)。
    """
    def __init__(self, in_channels, message_dim, num_groups=32):
        super().__init__()
        # 使用 affine=False，因为我们要自己预测参数
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=False)
        
        if message_dim > 0:
            # 线性层：将消息向量映射为 2 * channels (gamma + beta)
            self.embed = nn.Linear(message_dim, in_channels * 2)
            # 初始化：让初始状态接近标准的 Identity (gamma=1, beta=0)
            self.embed.weight.data[:, :in_channels].normal_(1, 0.02)
            self.embed.weight.data[:, in_channels:].zero_()
            self.embed.bias.data.zero_()
        else:
            self.embed = None
            
    def forward(self, x, message_vector):
        out = self.norm(x)
        
        if self.embed is not None and message_vector is not None:
            # 预测参数
            style = self.embed(message_vector) # [B, 2*C]
            gamma, beta = style.chunk(2, 1)    # [B, C], [B, C]
            
            # 调整维度以匹配特征图 [B, C, 1, 1]
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            
            out = out * gamma + beta
        return out

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResnetBlock(nn.Module):
    # 【修改 1】 __init__ 增加 message_dim 参数
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, message_dim=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # 【修改 2】 将 Normalize 替换为 ConditionalNorm
        self.norm1 = ConditionalNorm(in_channels, message_dim)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # 保留 temb 逻辑以兼容旧代码（如果不需要可以删掉）
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        
        # 【修改 3】 第二层也替换
        self.norm2 = ConditionalNorm(out_channels, message_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 【修改 4】 forward 增加 wm_emb 参数 (Watermark Embedding)
    def forward(self, x, temb=None, wm_emb=None):
        h = x
        # 传入水印向量
        h = self.norm1(h, wm_emb)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        # 传入水印向量
        h = self.norm2(h, wm_emb)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class VAEEncoder(nn.Module):
    def __init__(
        self, 
        *, 
        ch: int, 
        out_ch: int, 
        ch_mult: tuple[int, int, int, int] = (1,2,4,8), 
        num_res_blocks: int, 
        attn_resolutions: list[int], 
        dropout: float = 0.0,
        resamp_with_conv: bool = True, 
        in_channels: int, 
        resolution: int, 
        z_channels: int, 
        double_z: bool = True, 
        use_linear_attn: bool = False, 
        attn_type: str = "vanilla",
        message_dim: int = 0,  # 【修改 1】 新增参数
        **ignore_kwargs
    ) -> None:
        super().__init__()
        if use_linear_attn: 
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                # 【修改 2】 传入 message_dim
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, 
                                         temb_channels=self.temb_ch, dropout=dropout, 
                                         message_dim=message_dim))
                block_in = block_out
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        # 【修改 3】 Middle Block 也改
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, 
                                       temb_channels=self.temb_ch, dropout=dropout, 
                                       message_dim=message_dim)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, 
                                       temb_channels=self.temb_ch, dropout=dropout, 
                                       message_dim=message_dim)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, wm_emb=None):
        temb = None 

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # 【修改 5】 传入 wm_emb
                h = self.down[i_level].block[i_block](hs[-1], temb, wm_emb=wm_emb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, wm_emb=wm_emb) # 【修改 6】
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, wm_emb=wm_emb) # 【修改 7】

        # end
        h = self.norm_out(h) # 注意：这里的 norm_out 还是原来的 GroupNorm，如果需要也可以改成 Conditional
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(nn.Module):
    def __init__(
        self, 
        *, 
        ch: int,  
        out_ch: int, 
        ch_mult: tuple[int, int, int, int] = (1,2,4,8), 
        num_res_blocks: int, 
        attn_resolutions: list[int], 
        dropout: float = 0.0, 
        resamp_with_conv: bool = True, 
        in_channels: int, 
        resolution: int, 
        z_channels: int, 
        give_pre_end: bool = False, 
        tanh_out: bool = False, 
        use_linear_attn: bool = False, 
        attn_type: str = "vanilla", 
        bw: bool = False,
        message_dim: int = 0, # 【修改 1】
        **ignore_kwargs
    ) -> None:
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.bw = bw
        if self.bw:
            out_ch = 1

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        # 【修改 2】 传入 message_dim
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, 
                                       temb_channels=self.temb_ch, dropout=dropout, 
                                       message_dim=message_dim)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, 
                                       temb_channels=self.temb_ch, dropout=dropout, 
                                       message_dim=message_dim)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                # 【修改 3】 传入 message_dim
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, 
                                         temb_channels=self.temb_ch, dropout=dropout,
                                         message_dim=message_dim))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    # 【修改 4】 forward 增加 wm_emb
    def forward(self, z, wm_emb=None):
        temb = None

        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, wm_emb=wm_emb) # 【修改 5】
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, wm_emb=wm_emb) # 【修改 6】

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                # 【修改 7】
                h = self.up[i_level].block[i_block](h, temb, wm_emb=wm_emb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        if self.bw:
            h = h.repeat(1, 3, 1, 1)
        return h
