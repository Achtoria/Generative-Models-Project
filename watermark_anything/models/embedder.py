# --- START OF FILE embedder.py ---

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ..modules.vae import VAEEncoder, VAEDecoder
from ..modules.msg_processor import MsgProcessor


class Embedder(nn.Module):
    """
    Abstract class for watermark embedding.
    """
    def __init__(self) -> None:
        super(Embedder, self).__init__()
    
    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        """
        Generate a random message
        """
        return ...

    def get_last_layer(self) -> torch.Tensor:
        return None

    def forward(
        self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        return ...


class VAEEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    Modified for Vestigit-style injection (CBN).
    """
    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        msg_processor: MsgProcessor,
        message_dim: int = 128, # 【新增】CBN 向量的维度
        nbits: int = 32         # 【新增】输入消息的比特数
    ) -> None:
        super(VAEEmbedder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor # 保留它用于生成随机消息等工具函数
        
        # 【新增】消息编码器 (MLP)
        # 将 32 bit 的 0/1 消息 -> 128 维的稠密向量，用于 CBN
        self.msg_mlp = nn.Sequential(
            nn.Linear(nbits, 64),
            nn.SiLU(), # 或 ReLU
            nn.Linear(64, message_dim),
        )

    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.decoder.conv_out.weight
        return last_layer

    def forward(
        self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL.
        Returns:
            The watermarked images (Residual + Original).
        """
        # 1. 准备 CBN 需要的水印向量
        # msgs 通常是 [B, 32] 的 0/1 浮点数
        wm_emb = self.msg_mlp(msgs.float()) # -> [B, 128]

        # 2. 编码图像
        # 如果你希望 Encoder 也受水印影响，可以传入 wm_emb，否则传 None
        # Vestigit 主要是 Decoder 注入，但传给 Encoder 也没坏处，增加注入深度
        latents = self.encoder(imgs, wm_emb=wm_emb) 
        
        # 3. 【关键修改】不再使用 msg_processor 进行拼接
        
        # 4. 解码并注入水印
        # 直接把 latents 和 wm_emb 传给 decoder
        imgs_rec = self.decoder(latents, wm_emb=wm_emb)
        
        # Vestigit 论文通常是预测一个残差 (Residual) 加到原图上
        # 但如果你的 VAE 是训练来重建图像的 (Reconstruction)，直接返回 imgs_rec 也可以
        # 这里假设 VAE 学习的是重建带水印的图
        return imgs_rec


def build_embedder(name, cfg, nbits):
    if name.startswith('vae'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        # msg_processor 的 hidden_size 这里的设置对我们要用的 MLP 没影响，但也别删，防止报错
        cfg.msg_processor.hidden_size = nbits * 2 
        
        # 【关键修改 1】禁止通道扩张
        # 原代码：cfg.decoder.z_channels = (nbits * 2) + cfg.encoder.z_channels
        # 修改为：保持一致，因为我们不再拼接了
        cfg.decoder.z_channels = cfg.encoder.z_channels
        
        # 【关键修改 2】定义 CBN 向量维度
        cbn_dim = 128 # 这个维度要和 VAEEmbedder 里的定义一致
        
        # 【关键修改 3】把 message_dim 注入到 encoder 和 decoder 的配置中
        # 这样 vae.py 里的 __init__ 才能接收到 message_dim 参数
        cfg.encoder.message_dim = cbn_dim
        cfg.decoder.message_dim = cbn_dim

        # build the encoder, decoder and msg processor
        encoder = VAEEncoder(**cfg.encoder)
        msg_processor = MsgProcessor(**cfg.msg_processor)
        decoder = VAEDecoder(**cfg.decoder)
        
        # 【关键修改 4】实例化时传入 message_dim 和 nbits
        embedder = VAEEmbedder(encoder, decoder, msg_processor, message_dim=cbn_dim, nbits=nbits)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return embedder