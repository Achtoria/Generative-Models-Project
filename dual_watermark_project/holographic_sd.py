import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

def generate_holographic_image(
    prompt,
    dct_guidance,
    guidance_scale=100.0,
    guidance_target_val=5.0,
    num_inference_steps=50,
    guidance_interval=5, # 策略：每 5 步引导一次
    guidance_start_step=0,
    guidance_end_step=25, # 策略：只在前 50% 步数引导 (假设总步数50)
    device="cuda",
    seed=42
):
    # 1. 加载模型
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    # 使用 DDIM Scheduler，便于反演和引导
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # 冻结模型
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # 2. 初始化 Latents
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        generator=generator,
        device=device,
        dtype=torch.float16
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # 3. 编码 Prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, do_classifier_free_guidance=True, negative_prompt="")

    # 4. 去噪循环
    pipe.scheduler.set_timesteps(num_inference_steps)
    
    print(f"Generating with Holographic Guidance (Scale={guidance_scale}, Interval={guidance_interval})...")
    
    for i, t in enumerate(pipe.scheduler.timesteps):
        # --- 策略控制：判断当前步是否需要引导 ---
        # 条件1: 步数在指定窗口内 [start, end]
        is_in_window = (i >= guidance_start_step) and (i <= guidance_end_step)
        # 条件2: 符合间隔要求 (稀疏引导)
        is_interval_step = (i % guidance_interval == 0)
        
        should_guide = is_in_window and is_interval_step and (dct_guidance is not None)

        # 4.1 开启梯度 (仅当需要引导时)
        if should_guide:
            latents = latents.detach().requires_grad_(True)
            
            # 计算全息 Loss
            loss = dct_guidance.compute_loss(latents, target_val=guidance_target_val)
            
            # 计算梯度
            grad = torch.autograd.grad(loss, latents)[0]
            
            # 【修复点】增强引导力度
            # 方案 A: 移除 sigma_t，直接硬推 (Hard Guidance)
            # 这种方式对于"注入水印"这种非自然特征更有效
            latents = latents - guidance_scale * grad 
            
            # 引导后 detach
            latents = latents.detach()

        # 4.2 正常的 SD Step (不带梯度)
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # 5. 解码
    with torch.no_grad():
        image = pipe.decode_latents(latents)
        image = pipe.numpy_to_pil(image)[0]
        
    return image