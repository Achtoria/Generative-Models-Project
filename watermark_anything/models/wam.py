# --- START OF FILE wam.py ---

import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .embedder import Embedder
from .extractor import Extractor
from ..augmentation.augmenter import Augmenter
from ..modules.jnd import JND
# 【新增】确保路径正确，用于数值范围转换
from ..data.transforms import unnormalize_img, normalize_img 


class Wam(nn.Module):

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        augmenter: Augmenter,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        roll_probability: float = 0,
        img_size_extractor: int = 256,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.detector = detector
        self.augmenter = augmenter
        self.attenuation = attenuation
        self.scaling_w = scaling_w
        self.scaling_i = scaling_i
        self.roll_probability = roll_probability
        self.img_size_extractor = img_size_extractor

    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        return self.embedder.get_random_msg(bsz, nb_repetitions)

    def blend(self, imgs, preds_w) -> torch.Tensor:
        imgs_w = self.scaling_i * imgs + self.scaling_w * preds_w
        if self.attenuation is not None:
            imgs_w = self.attenuation(imgs, imgs_w)
        return imgs_w

    def forward(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        msgs: torch.Tensor = None,
        no_overlap: bool = False,
        nb_times: int = None,
        params = None,
        attack_module: nn.Module = None, # 【新增】攻击模块参数
    ) -> dict:
        
        if random.random() < self.roll_probability:
            roll = True
        else:
            roll = False
            
        h, w = params.img_size_extractor, params.img_size_extractor
        resize = transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BILINEAR)
        resize_nearest = transforms.Resize((h,w), interpolation=transforms.InterpolationMode.NEAREST)
        inverse_resize = transforms.Resize((imgs.shape[-2:]), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        
        # 处理 Masks
        aux = self.augmenter.mask_embedder(resize(imgs), masks=masks, no_overlap=no_overlap, nb_times=nb_times).to(imgs.device)
        if len(aux.shape)==4:
            aux = aux.unsqueeze(2)
        B, C, K, H, W = aux.shape
        aux = aux.view(B*K, C, H, W)
        aux = F.interpolate(aux, size=(imgs.shape[-2], imgs.shape[-1]), mode='nearest')
        mask_targets = aux.view(B, C, K, imgs.shape[-2], imgs.shape[-1])
        mask_targets = mask_targets.float()

        msgs_l = []
        combined_imgs = imgs.clone()
        
        # 1. 嵌入循环：生成并叠加所有水印
        for nb_wm in range(mask_targets.shape[1]):
            mask = mask_targets[:, nb_wm, :, :, :].float()
            
            msgs = self.get_random_msg(imgs.shape[0])
            msgs = msgs.to(imgs.device)
            msgs_l.append(msgs)
            
            # Embedder 生成 (CBN注入)
            deltas_w = self.embedder(resize(imgs), msgs)
            
            # 缩放并叠加
            imgs_w = self.scaling_i * imgs + self.scaling_w * inverse_resize(deltas_w)
            
            if self.attenuation is not None:
                imgs_w = self.attenuation(imgs, imgs_w)
            
            if not roll:
                combined_imgs = combined_imgs * (1 - mask) + imgs_w * mask
            else:
                combined_imgs = combined_imgs * torch.roll(1 - mask, shifts=-1, dims=0) + torch.roll(imgs_w, shifts=-1, dims=0) * torch.roll(mask, shifts=-1, dims=0)

        # -----------------------------------------------------------
        # 【核心修改】在此处插入 Differentiable JPEG 攻击
        # 此时 combined_imgs 已经是包含了所有水印的完整图像
        # -----------------------------------------------------------
        if attack_module is not None:
            # 1. 反归一化到 [0, 1]
            combined_imgs = unnormalize_img(combined_imgs)
            # 2. 执行攻击 (DiffJPEG)
            combined_imgs = attack_module(combined_imgs)
            # 3. 重新归一化回模型空间 (Standardized)
            combined_imgs = normalize_img(combined_imgs)
        # -----------------------------------------------------------

        # 2. 后处理增强 (几何变换等)
        if not roll:
            imgs_aug, mask_targets, selected_aug = self.augmenter.post_augment(combined_imgs, mask_targets.squeeze(2))
        else:
            imgs_aug, mask_targets, selected_aug = self.augmenter.post_augment(combined_imgs, torch.roll(mask_targets.squeeze(2), shifts=-1, dims=0))
        
        # 3. 检测
        preds = self.detector(resize(imgs_aug))
        
        msgs_l = torch.stack(msgs_l)
        msgs_l = msgs_l.transpose(0, 1)
        if roll:
            msgs_l = torch.roll(msgs_l, shifts=-1, dims=0)
            
        outputs = {
            "msgs": msgs_l,
            "masks": resize_nearest(mask_targets).bool(),
            # 注意：返回的 imgs_w 是最后一次迭代的结果，用于计算重建损失(LPIPS)。
            # 我们不应该返回被 JPEG 攻击过的图给 LPIPS，因为我们希望水印尽可能隐形。
            # 所以这里保持原样是正确的。检测器看到的是攻击后的 imgs_aug，感知损失看到的是干净的 imgs_w。
            "imgs_w": imgs_w if not roll else torch.roll(imgs_w, shifts=-1, dims=0),
            "imgs_aug": resize(imgs_aug),
            "preds": preds,
            "selected_aug": selected_aug,
        }
        return outputs

    # embed 和 detect 方法不需要修改，因为 inference 时通常手动控制攻击或不攻击
    def embed(self, imgs: torch.Tensor, msgs: torch.Tensor = None) -> dict:
        h, w = self.img_size_extractor, self.img_size_extractor
        resize = transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BILINEAR)
        inverse_resize = transforms.Resize((imgs.shape[-2:]), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)

        if msgs is None:
            msgs = self.get_random_msg(imgs.shape[0])

        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (h, w):
            imgs_res = resize(imgs)
        imgs_res = imgs_res.to(imgs.device)

        preds_w = self.embedder(imgs_res, msgs.to(imgs.device))

        if imgs.shape[-2:] != (h, w):
            preds_w = inverse_resize(preds_w)
        preds_w = preds_w.to(imgs.device)
        imgs_w = self.blend(imgs, preds_w)
        
        outputs = {
            "msgs": msgs,
            "preds_w": preds_w,
            "imgs_w": imgs_w,
        }
        return outputs

    def detect(self, imgs: torch.Tensor) -> dict:
        h, w = self.img_size_extractor, self.img_size_extractor
        resize = transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BILINEAR)

        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (h, w):
            imgs_res = resize(imgs)
        imgs_res = imgs_res.to(imgs.device)

        preds = self.detector(imgs_res).to(imgs.device)

        outputs = {
            "preds": preds,
        }
        return outputs