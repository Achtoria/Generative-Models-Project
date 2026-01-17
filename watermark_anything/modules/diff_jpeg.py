# 推荐直接使用 Kornia 库，或者把这个简单实现塞进 utils.py
import torch
import torch.nn as nn

# 如果不想装 kornia，这是一个极其简化的近似层（Rounding Approximation）
# 但最好还是去 pip install kornia
try:
    from kornia.enhance import jpeg_codec_differentiable
    class DiffJPEG(nn.Module):
        def __init__(self, quality=50):
            super().__init__()
            self.quality = quality
        def forward(self, x):
            # Kornia 的 JPEG 是可导的
            # 输入需要是 0-1 或 0-255，根据你的数据范围调整
            # quality 需要是形状为 [B] 的 Tensor（每个 batch 一个 quality 值）
            batch_size = x.shape[0]
            quality_tensor = torch.full((batch_size,), self.quality, dtype=torch.float32, device=x.device)
            return jpeg_codec_differentiable(x, quality_tensor)
except ImportError:
    print("Warning: Kornia not found. DiffJPEG will be skipped.")
    class DiffJPEG(nn.Module):
        def forward(self, x): return x