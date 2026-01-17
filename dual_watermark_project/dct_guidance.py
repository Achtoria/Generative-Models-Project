import torch
import torch.nn.functional as F
import math

class LatentDCTGuidance:
    def __init__(self, device, patch_size=8, target_channels=[0], target_freqs=[(3, 3)]):
        self.device = device
        self.patch_size = patch_size
        self.target_channels = target_channels
        self.target_freqs = target_freqs
        
        # 预计算 DCT 变换矩阵 (确保类型匹配)
        self.dct_matrix = self._get_dct_matrix(patch_size).to(device)

    def _get_dct_matrix(self, N):
        """生成标准 1D DCT-II 矩阵"""
        n = torch.arange(N).float()
        k = torch.arange(N).float()
        dct_matrix = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
        dct_matrix[0, :] *= 1.0 / math.sqrt(2)
        dct_matrix *= math.sqrt(2.0 / N)
        return dct_matrix

    def compute_loss(self, latents, target_val=5.0):
        """
        计算全息水印 Loss
        """
        # --- 类型对齐 ---
        dtype = latents.dtype
        dct_matrix = self.dct_matrix.to(dtype=dtype)
        
        # 1. Unfold: 切割 Patch
        patches = latents.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # 2. DCT 变换
        dct_coeffs = torch.matmul(dct_matrix, patches)
        dct_coeffs = torch.matmul(dct_coeffs, dct_matrix.t())
        
        # 3. 计算 Loss
        loss = 0.0
        count = 0
        
        # 准备 Target Tensor (广播到与 coeffs 相同的形状)
        # 这一步修复了 UserWarning
        # 我们需要知道 batch_size 等维度，但 F.mse_loss 支持自动广播，
        # 关键是 dtype 要一致，且最好是 Tensor
        target_tensor_scalar = torch.tensor(target_val, device=self.device, dtype=dtype)

        for ch in self.target_channels:
            for (u, v) in self.target_freqs:
                coeffs = dct_coeffs[:, ch, ..., u, v]
                
                # 显式扩展 target 到 coeffs 的形状 (消除 Warning 的关键)
                target_tensor = target_tensor_scalar.expand_as(coeffs)
                
                current_loss = F.mse_loss(coeffs, target_tensor)
                loss += current_loss
                count += 1
        
        return loss / max(count, 1)