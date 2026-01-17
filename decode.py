import argparse
import os
import torch
from PIL import Image
import torch.nn.functional as F
import sys

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

from notebooks.inference_utils import load_model_from_checkpoint, msg2str
from watermark_anything.data.metrics import msg_predict_inference
from watermark_anything.data.transforms import default_transform
from torchvision.utils import save_image

def decode(image_path, checkpoint_dir="checkpoints", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    json_path = os.path.join(checkpoint_dir, "params.json")
    ckpt_path = os.path.join(checkpoint_dir, "wam_mit.pth")
    
    if not os.path.exists(json_path):
        print(f"Error: params.json not found in {checkpoint_dir}")
        return
    if not os.path.exists(ckpt_path):
        print(f"Error: wam_mit.pth not found in {checkpoint_dir}")
        return

    print(f"Loading model from {checkpoint_dir}...")
    try:
        wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading image from {image_path}...")
    try:
        img = Image.open(image_path).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # Detect
    print("Detecting watermark...")
    with torch.no_grad():
        preds = wam.detect(img_pt)["preds"]
    
    # Post-processing
    mask_preds = F.sigmoid(preds[:, 0, :, :])
    bit_preds = preds[:, 1:, :, :]

    # Save mask
    mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)
    output_mask_path = f"{image_path}_mask_pred.png"
    save_image(mask_preds_res, output_mask_path)
    print(f"Predicted mask saved to {output_mask_path}")
    
    # Decode message
    # method='semihard' is default in msg_predict_inference, matching notebook usage implicitly or explicitly
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
    decoded_msg = msg2str(pred_message[0])
    
    print(f"\nDecoded Message: {decoded_msg}")
    return decoded_msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode watermark from an image using Watermark Anything Model")
    parser.add_argument("--image", type=str, default="/home/guantianrui/watermark-anything/assets/output.png", help="Path to the image file to decode")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Directory containing params.json and wam_mit.pth")
    
    args = parser.parse_args()
    
    decode(args.image, args.checkpoint)

