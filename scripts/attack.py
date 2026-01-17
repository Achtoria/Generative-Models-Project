import os
import sys
import click
import torch
from PIL import Image
from tqdm import tqdm
import dotenv

# Add current directory and parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adversarial"))

from distortions import apply_distortion, distortion_strength_paras
from adversarial.embedding import WarmupPGDEmbedding, EPS_FACTOR, ALPHA_FACTOR, N_STEPS, BATCH_SIZE
from feature_extractors import (
    ResNet18Embedding,
    VAEEmbedding,
    ClipEmbedding,
    KLVAEEmbedding,
)
from dev import (
    LIMIT,
    parse_image_dir_path,
    check_file_existence,
)

dotenv.load_dotenv()

def apply_adv_emb_attack(path, encoder, strength, target_path, limit, quiet, device="cuda:0"):
    # load embedding model
    if encoder == "resnet18":
        embedding_model = ResNet18Embedding("last")
    elif encoder == "clip":
        embedding_model = ClipEmbedding()
    elif encoder == "klvae8":
        embedding_model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
    elif encoder == "sdxlvae":
        embedding_model = VAEEmbedding("stabilityai/sdxl-vae")
    elif encoder == "klvae16":
        embedding_model = KLVAEEmbedding("kl-f16")
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")
    
    embedding_model = embedding_model.to(device)
    embedding_model.eval()
    
    # Create an instance of the attack
    # Strength in WAVES for adv-emb is usually 2, 4, 6, 8 (out of 255)
    # The relative strength 0.5 could map to 4.0? 
    # For resnet18 and clip, we enhance the strength as they were too weak (100% acc)
    multiplier = 32.0 if encoder in ["resnet18", "clip"] else 8.0
    eps = EPS_FACTOR * (strength * multiplier) 
    alpha = ALPHA_FACTOR * eps
    
    attack = WarmupPGDEmbedding(
        model=embedding_model,
        eps=eps,
        alpha=alpha,
        steps=N_STEPS,
        device=device,
    )

    indices = [i for i, exists in enumerate(check_file_existence(path, "{}.png", limit)) if exists]
    
    # Process in batches manually to avoid SimpleImageFolder overhead if we want consistency with other attacks
    # but using SimpleImageFolder from embedding.py is also fine.
    from adversarial.embedding import SimpleImageFolder
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.utils import save_image

    transform = transforms.ToTensor()
    # We need to filter SimpleImageFolder to only include the indices we want
    dataset = SimpleImageFolder(path, transform=transform)
    # Filter filenames by indices
    all_filenames = dataset.filenames
    dataset.filenames = [f for f in all_filenames if int(os.path.basename(f).split('.')[0]) in indices]

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device != "cpu")
    )

    for images, image_paths in tqdm(loader, desc=f"Adv-Emb {encoder}", disable=quiet):
        images = images.to(device)
        images_adv = attack.forward(images)
        for img_adv, image_path in zip(images_adv, image_paths):
            idx = os.path.basename(image_path)
            save_path = os.path.join(target_path, idx)
            save_image(img_adv, save_path)

@click.command()
@click.option("--path", "-p", required=True, help="Path to watermarked images (e.g., data/main/mscoco/wam)")
@click.option("--type", "-t", "attack_type", multiple=True, help="Type of attack (rotation, blur, adv-emb-resnet18, etc.)")
@click.option("--strength", "-s", type=float, default=0.5, help="Relative strength of the attack (0.0 to 1.0)")
@click.option("--limit", "-l", type=int, default=100, help="Limit number of images to attack")
@click.option("--quiet", "-q", is_flag=True, help="Quiet mode")
@click.option("--gpu", "-g", default="0", help="GPU ID")
def main(path, attack_type, strength, limit, quiet, gpu):
    """
    Apply lightweight distortions to watermarked images.
    Following WAVES native directory structure.
    """
    path = os.path.abspath(path)
    dataset_name, _, _, source_name = parse_image_dir_path(path, quiet=True)
    
    if not attack_type:
        # Default lightweight attacks if none specified
        attack_types = ["rotation", "blurring", "noise", "compression"]
    else:
        attack_types = list(attack_type)

    if not quiet:
        print(f"Applying attacks to {source_name} on {dataset_name}...")
        print(f"Selected attacks: {', '.join(attack_types)}")
        print(f"Relative strength: {strength}")

    # Base directory for attacked images
    attacked_base_root = os.path.join(os.environ.get("DATA_DIR"), "attacked", dataset_name)
    
    for atype in attack_types:
        if atype.startswith("adv-emb-"):
            encoder = atype.replace("adv-emb-", "")
            target_dir_name = f"{atype}-{strength}-{source_name}"
            target_path = os.path.join(attacked_base_root, target_dir_name)
            os.makedirs(target_path, exist_ok=True)
            
            if not quiet:
                print(f" -> Processing Adversarial {encoder} -> {target_path}")
            
            apply_adv_emb_attack(path, encoder, strength, target_path, limit, quiet, device=f"cuda:{gpu}")
            continue

        if atype not in distortion_strength_paras:
            print(f"Warning: Attack type '{atype}' not found in distortion_strength_paras. Skipping.")
            continue
            
        # WAVES naming convention: {attack_name}-{strength}-{source_name}
        # Note: WAVES often uses 'jpeg' for 'compression' in constants, but engine uses 'compression'
        display_name = atype if atype != "compression" else "jpeg"
        target_dir_name = f"{display_name}-{strength}-{source_name}"
        target_path = os.path.join(attacked_base_root, target_dir_name)
        os.makedirs(target_path, exist_ok=True)
        
        if not quiet:
            print(f" -> Processing {atype} -> {target_path}")
            
        # Get existing images
        indices = [i for i, exists in enumerate(check_file_existence(path, "{}.png", limit)) if exists]
        
        for idx in tqdm(indices, desc=f"Attacking {atype}", disable=quiet):
            img_path = os.path.join(path, f"{idx}.png")
            img = Image.open(img_path).convert("RGB")
            
            # apply_distortion can handle single image or list
            distorted_img = apply_distortion(
                [img], 
                atype, 
                strength=strength, 
                relative_strength=True,
                return_image=True
            )[0]
            
            distorted_img.save(os.path.join(target_path, f"{idx}.png"))

    if not quiet:
        print("All attacks completed.")

if __name__ == "__main__":
    main()
