import os
import sys
import click
import torch
import numpy as np
import onnxruntime as ort
import torch.multiprocessing as mp
from PIL import Image, ImageOps
import warnings
from tqdm.auto import tqdm
import dotenv

# Add watermark-anything directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "watermark-anything"))

try:
    from watermark_anything.data.metrics import msg_predict_inference
except ImportError:
    msg_predict_inference = None

from dev import (
    LIMIT,
    SUBSET_LIMIT,
    WATERMARK_METHODS,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
    encode_array_to_string,
)

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")


def get_indices(mode, path, quiet, subset, limit, subset_limit):
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).split("/")[-2],
        f"{str(path).split('/')[-1]}-decode.json",
    )
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        decoded_existences = [data[str(i)][mode] is not None for i in range(limit)]
        if (not subset and sum(decoded_existences) == limit) or (
            subset and sum(decoded_existences[:subset_limit]) == subset_limit
        ):
            return []
    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    reversed_latents_existences = check_file_existence(
        path, name_pattern="{}_reversed.pkl", limit=limit
    )
    if not quiet:
        print(
            f"Found {sum(image_existences)} images, and {sum(reversed_latents_existences)} reversed latents"
        )
    if mode == "tree_ring":
        existences = reversed_latents_existences
    elif mode in ["stable_sig", "stegastamp", "wam"]:
        existences = image_existences
    if not os.path.exists(json_path):
        indices = existence_to_indices(
            existences, limit=limit if not subset else subset_limit
        )
    else:
        indices = existence_to_indices(
            existence_operation(existences, decoded_existences, op="difference"),
            limit=limit if not subset else subset_limit,
        )
    return indices


def init_model(mode, gpu):
    if mode == "tree_ring":
        size = 64
        radius = 10
        channel = 3
        mask = torch.zeros((1, 4, size, size), dtype=torch.bool)
        x0 = y0 = size // 2
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        mask[:, channel] = torch.tensor(((x - x0) ** 2 + (y - y0) ** 2) <= radius**2)
        return mask
    elif mode == "stable_sig":
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        return ort.InferenceSession(
            os.path.join(os.environ.get("MODEL_DIR"), "stable_signature.onnx"),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(gpu)}],
            sess_options=session_options,
        )
    elif mode == "stegastamp":
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        return ort.InferenceSession(
            os.path.join(os.environ.get("MODEL_DIR"), "stega_stamp.onnx"),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(gpu)}],
            sess_options=session_options,
        )
    elif mode == "wam":
        from notebooks.inference_utils import load_model_from_checkpoint
        # You may need to adjust these paths or set them in .env
        wam_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "watermark-anything")
        json_path = os.path.join(wam_root, "checkpoints", "params.json")
        ckpt_path = os.path.join(wam_root, "checkpoints", "wam_mit.pth")
        
        # We need to temporarily change CWD because WAM uses relative paths for configs in params.json
        old_cwd = os.getcwd()
        os.chdir(wam_root)
        try:
            wam = load_model_from_checkpoint(json_path, ckpt_path)
        finally:
            os.chdir(old_cwd)
        return wam.to(f"cuda:{gpu}").eval()


def load_files(mode, path, indices):
    if mode == "tree_ring":
        return torch.cat(
            [
                torch.load(
                    os.path.join(path, f"{idx}_reversed.pkl"), map_location="cpu"
                )
                for idx in indices
            ],
            dim=0,
        )
    elif mode == "stable_sig":
        return np.stack(
            [
                (
                    (
                        np.array(
                            Image.open(os.path.join(path, f"{idx}.png")),
                            dtype=np.float32,
                        )
                        / 255.0
                        - [0.485, 0.456, 0.406]
                    )
                    / [0.229, 0.224, 0.225]
                )
                .transpose((2, 0, 1))
                .astype(np.float32)
                for idx in indices
            ],
            axis=0,
        )
    elif mode == "stegastamp":
        return np.stack(
            [
                np.array(
                    ImageOps.fit(
                        Image.open(os.path.join(path, f"{idx}.png")), (400, 400)
                    ),
                    dtype=np.float32,
                )
                / 255.0
                for idx in indices
            ],
            axis=0,
        )
    elif mode == "wam":
        from watermark_anything.data.transforms import default_transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            default_transform
        ])
        return torch.stack(
            [
                transform(Image.open(os.path.join(path, f"{idx}.png")).convert("RGB"))
                for idx in indices
            ],
            dim=0,
        )


def decode(mode, model, gpu, inputs):
    if mode == "tree_ring":
        fft_latents = torch.fft.fftshift(
            torch.fft.fft2(inputs.to(f"cuda:{gpu}")), dim=(-1, -2)
        )
        messages = torch.stack(
            [
                fft_latents[i].unsqueeze(0)[model].flatten()
                for i in range(fft_latents.shape[0])
            ],
            dim=0,
        )
        return torch.concatenate([messages.real, messages.imag], dim=1).cpu().numpy()
    elif mode == "stable_sig":
        outputs = model.run(
            None,
            {
                "image": inputs,
            },
        )
        return (outputs[0] > 0).astype(bool)
    elif mode == "stegastamp":
        outputs = model.run(
            None,
            {
                "image": inputs,
                "secret": np.zeros((inputs.shape[0], 100), dtype=np.float32),
            },
        )
        return outputs[2].astype(bool)
    elif mode == "wam":
        with torch.no_grad():
            outputs = model.detector(inputs.to(f"cuda:{gpu}"))
            # outputs shape: [B, 1+nbits, H, W]
            mask_preds = torch.sigmoid(outputs[:, 0:1, :, :])
            bit_preds = torch.sigmoid(outputs[:, 1:, :, :])
            if msg_predict_inference is not None:
                messages = msg_predict_inference(bit_preds, mask_preds)
                return messages.cpu().numpy()
            else:
                # Fallback implementation if import fails
                # Simple average over pixels where mask > 0.5
                bsz, nbits, h, w = bit_preds.size()
                masks = mask_preds > 0.5
                masks = masks.expand_as(bit_preds).bool()
                res = []
                for b in range(bsz):
                    b_bits = bit_preds[b]
                    b_mask = masks[b]
                    if b_mask.any():
                        m = b_bits[b_mask].view(nbits, -1).mean(dim=-1)
                    else:
                        m = b_bits.view(nbits, -1).mean(dim=-1)
                    res.append(m > 0.5)
                return torch.stack(res).cpu().numpy()


def worker(mode, gpu, path, indices, lock, counter, results):
    model = init_model(mode, gpu)
    batch_size = {"tree_ring": 32, "stable_sig": 4, "stegastamp": 4, "wam": 4}[mode]
    for it in range(0, len(indices), batch_size):
        inputs = load_files(
            mode, path, indices[it : min(it + batch_size, len(indices))]
        )
        messages = decode(mode, model, gpu, inputs)
        with lock:
            counter.value += inputs.shape[0]
            for idx, message in zip(
                indices[it : min(it + batch_size, len(indices))], messages
            ):
                results[idx] = encode_array_to_string(message)


def process(mode, indices, path, quiet):
    mp.set_start_method("spawn", force=True)  # Set start method to 'spawn'
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    num_workers = {
        "tree_ring": num_gpus,
        "stable_sig": num_gpus,
        "stegastamp": num_gpus * 2,
        "wam": num_gpus,
    }[mode]
    chunk_size = len(indices) // num_workers
    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        results = manager.dict()

        processes = []
        for rank in range(num_workers):
            start_idx = rank * chunk_size
            end_idx = None if rank == num_workers - 1 else (rank + 1) * chunk_size
            p = mp.Process(
                target=worker,
                args=(
                    mode,
                    rank % num_gpus,
                    path,
                    indices[start_idx:end_idx],
                    lock,
                    counter,
                    results,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(
            total=len(indices), desc="Decoding images or reversed latents", unit="file"
        ) as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= len(indices):
                        break

        for p in processes:
            p.join()

        return dict(results)


def report(mode, path, results, quiet, limit):
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).split("/")[-2],
        f"{str(path).split('/')[-1]}-decode.json",
    )
    if (not os.path.exists(json_path)) or (data := load_json(json_path)) is None:
        data = {}
        for idx in range(limit):
            data[str(idx)] = {
                _mode: results.get(idx) if mode == _mode else None
                for _mode in WATERMARK_METHODS.keys()
            }
    else:
        for idx, message in results.items():
            data[str(idx)][mode] = message
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    save_json(data, json_path)
    if not quiet:
        print(f"Decoded messages saved to {json_path}")


def single_mode(mode, path, quiet, subset, limit, subset_limit):
    if not quiet:
        print(f"Decoding {mode} messages")
    indices = get_indices(mode, path, quiet, subset, limit, subset_limit)
    if len(indices) == 0:
        if not quiet:
            print("All messages requested already decoded")
        return
    results = process(mode, indices, path, quiet)
    report(mode, path, results, quiet, limit)


@click.command()
@click.option(
    "--path", "-p", type=str, default=os.getcwd(), help="Path to image directory"
)
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
@click.option("--limit", "-l", type=int, default=LIMIT, help="Limit number of images")
@click.option(
    "--subset-limit",
    "-sl",
    type=int,
    default=SUBSET_LIMIT,
    help="Subset limit for image metrics",
)
def main(path, dry, subset, quiet, limit, subset_limit):
    _, _, _, source_name = parse_image_dir_path(path, quiet=quiet)
    # Reverse is only required for real and tree_ring watermarked images
    if source_name == "real":
        for mode in WATERMARK_METHODS.keys():
            single_mode(mode, path, quiet, subset, limit, subset_limit)
        return
    for mode in WATERMARK_METHODS.keys():
        if source_name.endswith(mode):
            single_mode(mode, path, quiet, subset, limit, subset_limit)
            return
    raise ValueError(f"Invalid source name {source_name} encountered")


if __name__ == "__main__":
    main()
