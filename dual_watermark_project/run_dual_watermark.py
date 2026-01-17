import os
import sys
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import omegaconf

# --- 1. 环境路径修正 ---
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

from dct_guidance import LatentDCTGuidance
from holographic_sd import generate_holographic_image

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 配置权重路径 ---
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    json_path = os.path.join(ckpt_dir, "params.json")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "wam_mit.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到权重文件。请检查 {ckpt_dir}")
        return

    # 加载配置
    print(f"Loading WAM config from {json_path}...")
    wam_params = omegaconf.OmegaConf.load(json_path)

    # --- Step 1: 生成全息潜空间水印图 ---
    print("\n>>> Step 1: Generating Image with Latent Hologram...")
    prompt = "A high quality photo of a cute cat sitting on a futuristic chair, 8k resolution"
    dct_guide = LatentDCTGuidance(device, target_channels=[0], target_freqs=[(3, 3)])
    
    img_pil_stage1 = generate_holographic_image(
        prompt,
        dct_guide,
        guidance_scale=0.2,      # 【调整】因为去掉了 sigma_t，这个值要改小！试 0.1 ~ 0.5
        guidance_target_val=10.0, # 【调整】目标值设大一点，抵抗 VAE 清洗
        guidance_interval=1,      # 【调整】改为每 1 步都引导，为了确保加上去
        guidance_end_step=50,     # 【调整】全程引导，不要停
        device=device
    )
    
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    img_pil_stage1.save(os.path.join(output_dir, "step1_latent_only.png"))
    
    # --- Step 2: 叠加 WAM ---
    print("\n>>> Step 2: Applying WAM (Pixel Watermark)...")
    print(f"Loading WAM from {ckpt_path}...")
    
    # 加载模型
    wam_raw = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    
    # 解包 DDP
    if hasattr(wam_raw, 'module'):
        wam = wam_raw.module
    else:
        wam = wam_raw
    
    # 打印属性以供调试 (如果再出错，这个信息很关键)
    # print(f"WAM Attributes: {[a for a in dir(wam) if not a.startswith('_')]}")
    
    img_tensor = default_transform(img_pil_stage1).unsqueeze(0).to(device)
    message = torch.randint(0, 2, (1, 32)).float().to(device)
    
    with torch.no_grad():
        # Step 2 既然通过了，说明 embedder 存在
        if hasattr(wam, 'embedder'):
            encoded_imgs = wam.embedder(img_tensor, message)
        else:
            # 备用方案：如果 embedder 也没有，打印错误
            print("CRITICAL: 'embedder' attribute missing. Dumping attributes:")
            print(dir(wam))
            return

    img_final_tensor = unnormalize_img(encoded_imgs)
    img_final_pil = TF.to_pil_image(img_final_tensor.squeeze(0).cpu().clamp(0, 1))
    img_final_pil.save(os.path.join(output_dir, "step2_dual_watermarked.png"))
    print("Dual watermarking complete.")

    # --- Step 3: 验证 ---
    print("\n>>> Step 3: Verifying Watermarks...")
    
    def check_wam(image_pil, target_msg, params):
        t_img = default_transform(image_pil).unsqueeze(0).to(device)
        
        # 构造 Dummy Masks
        # WAM forward 需要 masks 参数。通常给全 0 (无mask) 或全 1。
        # 这里的 mask 只是辅助，不会影响全局提取结果。
        masks = torch.ones(t_img.shape[0], 1, t_img.shape[2], t_img.shape[3]).to(device)
        
        with torch.no_grad():
            # 【终极修复】使用 forward + 关键字传参
            # 不再寻找 .extractor，直接走正门
            # params=params 确保参数传进去
            try:
                outputs = wam(t_img, masks, params=params)
            except TypeError as e:
                print(f"Warning: Keyword 'params' failed ({e}), trying positional...")
                # 最后的挣扎：位置参数
                outputs = wam(t_img, masks, params)
            
            # 解析输出
            if isinstance(outputs, dict):
                # 打印 keys 确认一下 (调试用)
                # print(f"Output keys: {outputs.keys()}")
                if 'preds_w' in outputs:
                    preds = outputs['preds_w']
                elif 'decoded' in outputs:
                    preds = outputs['decoded']
                else:
                    # 盲猜第一个
                    preds = list(outputs.values())[0]
            else:
                preds = outputs

            # 转 Float (防止 Long 类型报错)
            preds = preds.float()

            # 空间平均
            if preds.dim() > 2:
                preds = preds.mean(dim=(-1, -2))
                
            # 计算 Acc
            pred_bits = (torch.sigmoid(preds) > 0.5).float()
            acc = (pred_bits == target_msg).float().mean().item()
            
        return acc

    def check_latent_hologram(image_pil):
        from diffusers import StableDiffusionPipeline
        temp_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
        t_img = TF.to_tensor(image_pil).unsqueeze(0).to(device)
        norm_img = (t_img * 2.0 - 1.0)
        with torch.no_grad():
            latents = temp_pipe.vae.encode(norm_img).latent_dist.sample() * 0.18215
        loss = dct_guide.compute_loss(latents, target_val=5.0)
        return loss.item()

    # 原图测试
    print("Checking Original Image...")
    acc_wam = check_wam(img_final_pil, message, wam_params)
    loss_latent = check_latent_hologram(img_final_pil)
    print(f"[Original] WAM Acc: {acc_wam:.2%} | Latent Loss: {loss_latent:.4f}")

    # --- 模拟 VAE 攻击 ---
    print("Simulating VAE Attack (AI Regeneration)...")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    
    t_img = TF.to_tensor(img_final_pil).unsqueeze(0).to(device)
    norm_img = (t_img * 2.0 - 1.0)
    
    with torch.no_grad():
        latents_attacked = pipe.vae.encode(norm_img).latent_dist.sample() * 0.18215
        recons = pipe.vae.decode(latents_attacked / 0.18215).sample
        
        # 【修复】Tensor -> Numpy -> PIL
        # 1. 反归一化
        recons = (recons / 2 + 0.5).clamp(0, 1)
        # 2. 移到 CPU 并转 Numpy
        recons_np = recons.cpu().permute(0, 2, 3, 1).float().numpy()
        
    # 3. 传给 numpy_to_pil
    recons_pil = pipe.numpy_to_pil(recons_np)[0]
    recons_pil.save(os.path.join(output_dir, "attacked_vae.png"))
    
    # 攻击后测试
    print("Checking Attacked Image...")
    acc_wam_att = check_wam(recons_pil, message, wam_params)
    loss_latent_att = check_latent_hologram(recons_pil)
    print(f"[VAE Attack] WAM Acc: {acc_wam_att:.2%} | Latent Loss: {loss_latent_att:.4f}")
    
    # 最终结论
    print("\n--- Final Analysis ---")
    if acc_wam_att < 0.8:
        print(">> WAM Status: DESTROYED (As expected)")
    else:
        print(">> WAM Status: SURVIVED (Unexpected!)")
        
    if loss_latent_att < 10.0: # 这里的阈值取决于 baseline
        print(">> Latent Status: SURVIVED (Success!)")
    else:
        print(">> Latent Status: WEAK (Try increasing guidance_scale)")

if __name__ == "__main__":
    run_experiment()