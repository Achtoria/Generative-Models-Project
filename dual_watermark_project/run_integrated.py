import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import omegaconf
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torchvision.transforms.functional as TF

# ==========================================
# 1. 环境与路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)

os.chdir(root_dir)

try:
    from notebooks.inference_utils import (
        load_model_from_checkpoint, 
        default_transform, 
        unnormalize_img
    )
except ImportError as e:
    print(f"Error importing inference_utils: {e}")
    raise e

# ==========================================
# 2. 全息水印工具类 (用于验证)
# ==========================================
class LatentDCTGuidance:
    def __init__(self, device, patch_size=8, target_channels=[0], target_freqs=[(0, 1)], image_size=64):
        self.device = device
        self.patch_size = patch_size
        self.target_channels = target_channels
        self.target_freqs = target_freqs
        
        self.dct_matrix = self._get_dct_matrix(patch_size).to(device)
        
        # 这里的 Seed 必须和训练时完全一致 (12345)
        grid_size = image_size // patch_size
        generator = torch.Generator(device).manual_seed(12345) 
        self.secret_key = torch.randint(0, 2, (1, grid_size, grid_size), generator=generator, device=device).float()
        self.secret_key = self.secret_key * 2 - 1 

    def _get_dct_matrix(self, N):
        n = torch.arange(N).float()
        k = torch.arange(N).float()
        dct_matrix = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
        dct_matrix[0, :] *= 1.0 / math.sqrt(2)
        dct_matrix *= math.sqrt(2.0 / N)
        return dct_matrix

    def compute_accuracy(self, latents):
        # 自动适配尺寸 (如果输入不是 64x64 Latent，比如来自 256px 图片的 32x32 Latent)
        if latents.shape[-1] != 64:
            latents = F.interpolate(latents, size=(64, 64), mode='nearest')

        dtype = latents.dtype
        dct_matrix = self.dct_matrix.to(dtype=dtype)
        secret_key = self.secret_key.to(dtype=dtype)
        
        patches = latents.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        dct_coeffs = torch.matmul(dct_matrix, patches)
        dct_coeffs = torch.matmul(dct_coeffs, dct_matrix.t())
        
        correct = 0
        total = 0
        for ch in self.target_channels:
            for (u, v) in self.target_freqs:
                coeffs = dct_coeffs[:, ch, ..., u, v]
                key = secret_key.expand_as(coeffs)
                # 符号一致即匹配
                correct += ((coeffs * key) > 0).float().sum()
                total += coeffs.numel()
        return correct / total

# ==========================================
# 3. 主程序 (LoRA 推理 + WAM)
# ==========================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # WAM 权重路径
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    json_path = os.path.join(ckpt_dir, "params.json")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    if not os.path.exists(ckpt_path): ckpt_path = os.path.join(ckpt_dir, "wam_mit.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: WAM checkpoint not found at {ckpt_path}")
        return
    
    # LoRA 权重路径 (你刚刚训练出来的)
    lora_path = os.path.join(current_dir, "lora_coco_watermark")
    if not os.path.exists(lora_path):
        print(f"Error: LoRA path not found at {lora_path}. Please run train_coco_watermark.py first.")
        return

    print("Loading WAM Params...")
    wam_params = omegaconf.OmegaConf.load(json_path)

    # 初始化 SD Pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    # 使用标准调度器
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # 初始化 Latent 检测工具 (仅用于验证)
    dct_guide = LatentDCTGuidance(device, target_channels=[0], target_freqs=[(0, 1)], image_size=64)
    
    prompt = "A high quality photo of a cute cat sitting on a futuristic chair, 8k resolution"
    seed = 42
    out_dir = os.path.join(current_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PART 1: 生成过程 (Reference + LoRA)
    # ---------------------------------------------------------
    print("\n>>> [Step 1] Generating Images...")

    # 1.A 生成【参考图】(Clean Reference - 无 LoRA)
    # 这张图用于计算完美的 WAM JND Mask
    generator = torch.Generator(device).manual_seed(seed)
    print("Generating Clean Reference (No LoRA)...")
    img_ref = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
    img_ref.save(os.path.join(out_dir, "ref_clean.png"))

    # 1.B 生成【全息底图】(Holographic Latent - 加载 LoRA)
    # 这张图里已经内嵌了抗 VAE 的 Latent 水印
    print(f"Loading LoRA from {lora_path}...")
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    
    generator = torch.Generator(device).manual_seed(seed) # 必须用相同 Seed 保证构图一致
    print("Generating Holographic Latent Image (With LoRA)...")
    img_latent = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
    img_latent.save(os.path.join(out_dir, "step1_latent_lora.png"))

    # ---------------------------------------------------------
    # PART 2: 移花接木 (Reference-based WAM Embedding)
    # ---------------------------------------------------------
    print("\n>>> [Step 2] Applying WAM (Reference Transfer)...")
    
    wam_raw = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    if hasattr(wam_raw, 'module'):
        wam = wam_raw.module
    else:
        wam = wam_raw
    
    # Resize 到 256 (WAM 舒适区)
    img_ref_256 = img_ref.resize((256, 256), Image.BICUBIC)
    img_latent_256 = img_latent.resize((256, 256), Image.BICUBIC)
    
    tensor_ref = default_transform(img_ref_256).unsqueeze(0).to(device)
    tensor_latent = default_transform(img_latent_256).unsqueeze(0).to(device)
    
    message = torch.randint(0, 2, (1, 32)).float().to(device)
    masks = torch.ones(1, 1, 256, 256).to(device)

    with torch.no_grad():
        # A. 在【干净参考图】上打 WAM 水印
        # 因为参考图没有 LoRA 带来的微小纹理，JND 算法会完美工作
        try:
            outputs = wam(tensor_ref, masks, message, params=wam_params)
        except TypeError:
            outputs = wam(tensor_ref, masks, message, wam_params)
        
        if isinstance(outputs, dict) and 'imgs_w' in outputs:
            encoded_ref = outputs['imgs_w']
        else:
            print("CRITICAL: Failed to get watermarked image from WAM.")
            return

        # B. 提取【完美残差】
        residual = encoded_ref - tensor_ref
        
        # C. 叠加到【LoRA 图】上
        final_tensor = tensor_latent + residual

    img_final_tensor = unnormalize_img(final_tensor)
    img_final_pil = TF.to_pil_image(img_final_tensor.squeeze(0).cpu().clamp(0, 1))
    
    img_final_pil.save(os.path.join(out_dir, "step2_dual.png"))
    target_image = img_final_pil
    
    print("Dual watermark applied successfully.")

    # ---------------------------------------------------------
    # PART 3: 验证与攻击
    # ---------------------------------------------------------
    print("\n>>> [Step 3] Verification & Attack Simulation...")

    def detect_wam(img_pil, target_msg):
        t_img = default_transform(img_pil).unsqueeze(0).to(device)
        masks = torch.ones(t_img.shape[0], 1, t_img.shape[2], t_img.shape[3]).to(device)
        with torch.no_grad():
            try:
                outputs = wam(t_img, masks, params=wam_params)
            except TypeError:
                outputs = wam(t_img, masks, wam_params)
            
            if isinstance(outputs, dict):
                preds = outputs.get('preds_w', list(outputs.values())[0])
            else:
                preds = outputs
            
            preds = preds.float()
            if preds.dim() > 2: preds = preds.mean(dim=(-1, -2))
            pred_bits = (torch.sigmoid(preds) > 0.5).float()
            acc = (pred_bits == target_msg).float().mean().item()
        return acc

    def detect_latent(img_pil):
        # 无论图片多大，VAE 都能处理
        # 但如果是 256 的图，Latent 就是 32x32，工具类会自动 interpolate 到 64 来匹配 Key
        t_img = TF.to_tensor(img_pil).unsqueeze(0).to(device)
        norm_img = (t_img * 2.0 - 1.0)
        with torch.no_grad():
            lats = pipe.vae.encode(norm_img).latent_dist.sample() * 0.18215
        return dct_guide.compute_accuracy(lats).item()

    # 1. 检测原图
    acc_wam = detect_wam(target_image, message)
    acc_latent = detect_latent(target_image)
    print(f"Original Image (Dual Watermarked) -> WAM Acc: {acc_wam:.2%} | Latent Key Match: {acc_latent:.2%}")

    # 2. 模拟 VAE 攻击
    print("Simulating VAE Attack (AI Regeneration)...")
    with torch.no_grad():
        t_img = TF.to_tensor(target_image).unsqueeze(0).to(device)
        norm_img = (t_img * 2.0 - 1.0)
        latents_attacked = pipe.vae.encode(norm_img).latent_dist.sample() * 0.18215
        recons = pipe.vae.decode(latents_attacked / 0.18215).sample
        
        recons = (recons / 2 + 0.5).clamp(0, 1)
        recons_np = recons.cpu().permute(0, 2, 3, 1).float().numpy()
        recons_pil = pipe.numpy_to_pil(recons_np)[0]
    recons_pil.save(os.path.join(out_dir, "attacked_vae.png"))

    # 3. 检测攻击后
    acc_wam_att = detect_wam(recons_pil, message)
    acc_latent_att = detect_latent(recons_pil)
    print(f"Attacked Image -> WAM Acc: {acc_wam_att:.2%} | Latent Key Match: {acc_latent_att:.2%}")

    # --- 最终判定 ---
    print("\n=== Final Report ===")
    
    # 期望：WAM 死亡 (<0.8), Latent 存活 (>0.55)
    # 因为 LoRA 是训练出来的，Latent 水印应该比之前 Guidance 的更稳健
    
    if acc_wam_att > 0.8:
        print("[FAIL] WAM unexpectedly survived VAE attack.")
    else:
        print("[PASS] WAM was successfully washed out by VAE.")
        
    if acc_latent_att > 0.55:
        print(f"[PASS] Latent Hologram SURVIVED! Match Rate ({acc_latent_att:.2%}) > Random.")
        print(">>> CONCLUSION: AI Regeneration Detected (Success).")
    else:
        print(f"[FAIL] Latent Hologram died ({acc_latent_att:.2%}).")
        print("Possible reasons: LoRA training steps too few? Try training for 2000+ steps.")

if __name__ == "__main__":
    run_experiment()