import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import click
import dotenv
from tabulate import tabulate

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adversarial"))

from distortions import apply_distortion, distortion_strength_paras
from dev import (
    LIMIT,
    parse_image_dir_path,
    check_file_existence,
    decode_array_from_string,
    GROUND_TRUTH_MESSAGES,
)
from dev.eval import bit_error_rate
from metrics.image import compute_psnr
from scripts.decode import load_files, decode as waves_decode, init_model
from regeneration.regen import regen_diff, regen_vae

dotenv.load_dotenv()

# --- Attack Configuration ---
# Set "on" to include in the default evaluation, "off" to skip.
ATTACK_CONFIG = {
    # 8 Standard Distortion Attacks
    "rotation": "on",
    "resizedcrop": "on",
    "erasing": "on",
    "brightness": "on",
    "contrast": "on",
    "blurring": "on",
    "noise": "on",
    "compression": "on",
    
    # 4 Combination Attacks (WAVES Native)
    "combo-geometric": "off",    # rotation + resizedcrop
    "combo-photometric": "off",  # brightness + contrast
    "combo-degradation": "off",  # blurring + noise + compression
    "combo-all": "off",          # all the above
    
    # 3 Primary Adversarial Attacks (Embedding Space)
    "adv-emb-resnet18": "on",
    "adv-emb-clip": "on",
    "adv-emb-klvae8": "on",
    
    # Representative Regeneration/Diffusion Attacks
    "regen-diffusion": "on",     # Stable-Diffusion (SD 1.4) - ON as representative
    "regen-vae": "on",           # VAE-based compression - ON as representative    "regen-diffusion-prompt": "off", # Diffusion with prompt guidance
    "regen-2x-diffusion": "off",     # Double diffusion pass
    "regen-4x-diffusion": "off",     # Quadruple diffusion pass    
    # Advanced Adversarial Variations (WAVES/Adversarial)
    "adv-emb-sdxlvae": "off",
    "adv-emb-klvae16": "off",
    "adv-cls-unwm-wm": "off",    # Classifier-based surrogate attack (requires .pth)
    "adv-cls-real-wm": "off",    # Classifier-based surrogate attack (requires .pth)
}

@click.command()
@click.option("--path", "-p", required=True, help="Path to watermarked images (e.g., data/main/mscoco/wam)")
@click.option("--method", "-m", default="wam", help="Watermarking method to evaluate")
@click.option("--strength", "-s", type=float, default=0.5, help="Relative strength of the attacks (0.0 to 1.0)")
@click.option("--limit", "-l", type=int, default=100, help="Number of images to evaluate")
@click.option("--gpu", "-g", type=int, default=0, help="GPU ID to use")
@click.option("--attack", "-t", multiple=True, help="Override default config and run specific attacks.")
def main(path, method, strength, limit, gpu, attack):
    """
    Unified evaluation script: Attack -> Decode -> Verify Accuracy.
    """
    path = os.path.abspath(path)
    dataset_name, _, _, source_name = parse_image_dir_path(path, quiet=True)
    gt_msg = GROUND_TRUTH_MESSAGES.get(method)
    
    if gt_msg is None:
        raise ValueError(f"Ground truth message for method '{method}' not found in dev/constants.py")

    # Determine which attacks to run
    if attack:
        attack_types = list(attack)
    else:
        attack_types = [atype for atype, status in ATTACK_CONFIG.items() if status == "on"]
    
    results = []
    
    # Path to original (unwatermarked) images for PSNR calculation
    original_path = os.path.join(os.environ.get("DATA_DIR"), "main", dataset_name, "real")
    
    print(f"\n--- Starting Robustness Evaluation for {method.upper()} ---")
    print(f"Dataset: {dataset_name} | Images: {limit} | Strength: {strength}")
    print(f"Watermarked Path: {path}")
    print(f"Original Path:    {original_path}")
    print(f"Active Attacks: {', '.join(attack_types)}")
    
    # 0. Initialize Model
    print(f"Loading {method} decoder model...")
    model = init_model(method, gpu)
    
    # 1. Evaluate Clean Accuracy first
    print("\n[Step 1/3] Evaluating Clean Accuracy...")
    indices = [i for i, exists in enumerate(check_file_existence(path, "{}.png", limit)) if exists]
    if not indices:
        print("Error: No watermarked images found at the provided path.")
        return

    clean_inputs = load_files(method, path, indices)
    clean_decoded = waves_decode(method, model, gpu, clean_inputs)
    clean_bers = [bit_error_rate(msg, gt_msg) for msg in clean_decoded]
    clean_acc = 1 - np.mean(clean_bers)

    # Calculate Clean PSNR
    clean_psnrs = []
    for idx in tqdm(indices, desc="Calculating Clean PSNR", leave=False):
        img_w = Image.open(os.path.join(path, f"{idx}.png")).convert("RGB")
        img_o = Image.open(os.path.join(original_path, f"{idx}.png")).convert("RGB")
        clean_psnrs.append(compute_psnr(img_w, img_o))
    clean_psnr = np.mean(clean_psnrs)

    results.append(["None (Clean)", strength, f"{clean_acc*100:.2f}%", f"{clean_psnr:.2f}"])
    print(f"Clean Bit Accuracy: {clean_acc*100:.2f}% | PSNR: {clean_psnr:.2f}")

    # 2. Loop through attacks
    print("\n[Step 2/3] Running Attacks and Decoding...")
    attacked_base_root = os.path.join(os.environ.get("DATA_DIR"), "attacked", dataset_name)
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    
    # Import adversarial helper from scripts.attack
    from scripts.attack import apply_adv_emb_attack

    for atype in attack_types:
        # Determine directory name
        display_name = atype.replace("adv-emb-", "adv-").replace("regen-", "regen-")
        if display_name == "compression": display_name = "jpeg"
            
        target_dir_name = f"{display_name}-{strength}-{source_name}"
        target_path = os.path.join(attacked_base_root, target_dir_name)
        os.makedirs(target_path, exist_ok=True)
        
        print(f"\nProcessing Attack: {atype}...")
        
        # Branch 0: Combination Attacks
        if atype.startswith("combo-"):
            if atype == "combo-geometric":
                sub_attacks = ["rotation", "resizedcrop"]
            elif atype == "combo-photometric":
                sub_attacks = ["brightness", "contrast"]
            elif atype == "combo-degradation":
                sub_attacks = ["blurring", "noise", "compression"]
            else: # combo-all
                sub_attacks = ["rotation", "resizedcrop", "erasing", "brightness", "contrast", "blurring", "noise", "compression"]
            
            for idx in tqdm(indices, desc=f"Applying {atype}", leave=False):
                img = Image.open(os.path.join(path, f"{idx}.png")).convert("RGB")
                for sub in sub_attacks:
                    img = apply_distortion([img], sub, strength=strength, relative_strength=True)[0]
                img.save(os.path.join(target_path, f"{idx}.png"))

        # Branch 1: Adversarial Embedding Attacks
        elif atype.startswith("adv-emb-"):
            encoder = atype.replace("adv-emb-", "")
            apply_adv_emb_attack(path, encoder, strength, target_path, limit, quiet=False, device=f"cuda:{gpu}")
        
        # Branch 2: Regeneration/Diffusion Attacks
        elif atype == "regen-diffusion":
            # For regen-diffusion, strength is typically noise steps (e.g., 0.5 * 100 = 50 steps)
            noise_steps = int(strength * 100) 
            for idx in tqdm(indices, desc="Applying Diffusion Regen", leave=False):
                img_path = os.path.join(path, f"{idx}.png")
                img = Image.open(img_path).convert("RGB")
                # Using SD 1.4 by default
                attacked_img = regen_diff(img, noise_steps, model="CompVis/stable-diffusion-v1-4", device=device)
                attacked_img.save(os.path.join(target_path, f"{idx}.png"))

        elif atype.endswith("-diffusion"): # 2x or 4x
            from regeneration.regen import rinse_2xDiff, rinse_4xDiff
            noise_steps = int(strength * 100)
            func = rinse_2xDiff if "2x" in atype else rinse_4xDiff
            for idx in tqdm(indices, desc=f"Applying {atype}", leave=False):
                img = Image.open(os.path.join(path, f"{idx}.png")).convert("RGB")
                attacked_img = func(img, noise_steps, model="CompVis/stable-diffusion-v1-4", device=device)
                attacked_img.save(os.path.join(target_path, f"{idx}.png"))
                
        elif atype == "regen-vae":
            # For regen-vae, strength is quality (1-8). WAVES strength 0.5 -> 4
            quality = max(1, int(strength * 8))
            for idx in tqdm(indices, desc="Applying VAE Regen", leave=False):
                img_path = os.path.join(path, f"{idx}.png")
                img = Image.open(img_path).convert("RGB")
                attacked_img = regen_vae(img, quality, model="bmshj2018-factorized", device=device)
                attacked_img.save(os.path.join(target_path, f"{idx}.png"))

        # Branch 3: Standard Distortions
        else:
            for idx in tqdm(indices, desc="Applying Distortion", leave=False):
                img_path = os.path.join(path, f"{idx}.png")
                img = Image.open(img_path).convert("RGB")
                distorted_img = apply_distortion([img], atype, strength=strength, relative_strength=True)[0]
                distorted_img.save(os.path.join(target_path, f"{idx}.png"))
        
        # Decode and Verify
        attacked_inputs = load_files(method, target_path, indices)
        decoded = waves_decode(method, model, gpu, attacked_inputs)
        bers = [bit_error_rate(msg, gt_msg) for msg in decoded]
        acc = 1 - np.mean(bers)
        
        # Calculate Attacked PSNR
        attack_psnrs = []
        for idx in tqdm(indices, desc="Calculating Attack PSNR", leave=False):
            img_a_path = os.path.join(target_path, f"{idx}.png")
            img_o_path = os.path.join(original_path, f"{idx}.png")
            
            img_a = Image.open(img_a_path).convert("RGB")
            img_o = Image.open(img_o_path).convert("RGB")
            
            # Auto-resize if diffusion/VAE changed dimensions
            if img_a.size != img_o.size:
                img_a = img_a.resize(img_o.size, Image.BILINEAR)
                
            attack_psnrs.append(compute_psnr(img_a, img_o))
        at_psnr = np.mean(attack_psnrs)

        results.append([atype, strength, f"{acc*100:.2f}%", f"{at_psnr:.2f}"])
        print(f" -> Bit Accuracy: {acc*100:.2f}% | PSNR: {at_psnr:.2f}")

    # 3. Final Report
    print("\n[Step 3/3] Final Robustness Report")
    print(tabulate(results, headers=["Attack Type", "Strength", "Bit Accuracy", "PSNR"], tablefmt="grid"))
    
    # Optional: Save results to JSON
    # (Leaving this out for brevity unless requested)

if __name__ == "__main__":
    main()
