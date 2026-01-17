import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm.auto import tqdm
import math

# --- 1. DCT 工具 (保持不变) ---
class LatentDCTGuidance:
    def __init__(self, device, patch_size=8, target_channels=[0], target_freqs=[(0, 1)], image_size=64):
        self.device = device
        self.patch_size = patch_size
        self.target_channels = target_channels
        self.target_freqs = target_freqs
        self.dct_matrix = self._get_dct_matrix(patch_size).to(device)
        
        # 固定 Key
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

    def compute_loss(self, latents, target_strength=0.5): # 【修正】默认改为 0.5
        dtype = latents.dtype
        dct_matrix = self.dct_matrix.to(dtype=dtype)
        secret_key = self.secret_key.to(dtype=dtype)
        
        patches = latents.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        dct_coeffs = torch.matmul(dct_matrix, patches)
        dct_coeffs = torch.matmul(dct_coeffs, dct_matrix.t())
        
        loss = 0.0
        count = 0
        
        for ch in self.target_channels:
            for (u, v) in self.target_freqs:
                coeffs = dct_coeffs[:, ch, ..., u, v]
                target = secret_key * target_strength
                target = target.expand_as(coeffs)
                
                # 使用 MSE 强力注入
                current_loss = F.mse_loss(coeffs, target)
                loss += current_loss
                count += 1
        return loss / max(count, 1)

# --- 2. 健壮的 COCO 数据加载器 ---
class SimpleImageDataset(Dataset):
    def __init__(self, img_dir, tokenizer, size=512):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.size = size
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = []
        # 递归搜索，防止图片在子文件夹
        for root, _, files in os.walk(img_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    self.image_paths.append(os.path.join(root, f))
        
        print(f"Found {len(self.image_paths)} images in {img_dir}")
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {img_dir}. Check path.")

        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # 这里的 convert("RGB") 非常关键，防止黑白图报错
            image = Image.open(self.image_paths[idx]).convert("RGB")
            pixel_values = self.transforms(image)
        except Exception as e:
            print(f"Warning: Skipping bad image {self.image_paths[idx]}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

        # 使用通用 Prompt，专注于学习纹理而非语义绑定
        prompt = "a high quality photo" 
        
        inputs = self.tokenizer(
            prompt, 
            padding="max_length", 
            truncation=True, 
            max_length=self.tokenizer.model_max_length, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0)
        }

# --- 3. 训练主程序 ---
def train(coco_path):
    device = "cuda"
    model_id = "runwayml/stable-diffusion-v1-5"
    output_dir = "lora_coco_watermark"
    
    # 显存优化：Batch size 设小点，积累梯度
    train_batch_size = 2 
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    max_train_steps = 1500 # 1500步足够让 LoRA 学会纹理了
    
    print("Loading SD Components...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    text_encoder = pipe.text_encoder.to(device)
    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    print("Injecting LoRA...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    dataset = SimpleImageDataset(coco_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    
    # 【参数调整】使用 0.5 强度，更温和
    dct_guide = LatentDCTGuidance(device, target_channels=[0], target_freqs=[(0, 1)])
    target_strength = 0.5
    
    print(f"Starting Training for {max_train_steps} steps...")
    global_step = 0
    progress_bar = tqdm(total=max_train_steps)
    
    # 无限循环 DataLoader (直到步数满足)
    epoch = 0
    while global_step < max_train_steps:
        epoch += 1
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # 1. 真实图片的 Latent (Ground Truth x0)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215 # SD 标准缩放因子
                
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            # 2. 加噪
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 3. 预测
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # 4. Loss 计算
            # A. 重建 Loss
            loss_diff = F.mse_loss(noise_pred, noise, reduction="mean")
            
            # B. 水印 Loss
            # 反推 x0
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
            sqrt_alpha_prod = alphas_cumprod[timesteps].flatten().view(bsz, 1, 1, 1) ** 0.5
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).flatten().view(bsz, 1, 1, 1) ** 0.5
            pred_x0 = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
            
            loss_wm = dct_guide.compute_loss(pred_x0, target_strength=target_strength)
            
            # 【权重调整】
            # loss_diff 通常在 0.05-0.1 之间
            # loss_wm (MSE) 也在这个量级
            # 给 0.1 的权重比较合适，既不破坏画质，又能注入水印
            loss = loss_diff + 0.1 * loss_wm
            
            loss.backward()
            
            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 10 == 0:
                    progress_bar.set_description(f"Loss: {loss.item():.4f} | Diff: {loss_diff.item():.4f} | WM: {loss_wm.item():.4f}")
            
            if global_step >= max_train_steps:
                break
            
    unet.save_pretrained(output_dir)
    print(f"Training finished. LoRA saved to {output_dir}")

if __name__ == "__main__":
    COCO_IMG_DIR = "/share_data/guantianrui/datasets/COCO/train2017" 
    
    if not os.path.exists(COCO_IMG_DIR):
        print(f"Path {COCO_IMG_DIR} not found. Please edit the script.")
    else:
        train(COCO_IMG_DIR)