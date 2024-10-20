from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from model import ControlLDM, Diffusion, RRDBNet
from utils.common import (instantiate_from_config, wavelet_decomposition,
                          wavelet_reconstruction)
from utils.helpers import pad_to_multiples_of
from utils.inference import load_model_from_url
from utils.sampler import SpacedSampler


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--upscale", type=float, required=True)
    ### input parameters
    parser.add_argument("-i", "--input", type=str, required=True, help="input image path")
    parser.add_argument("-o", "--output", type=str, default="results", help="output folder")
    ### sampling parameters
    parser.add_argument("--steps", type=int, default=50, help="# of steps for denosing")
    parser.add_argument("--pos_prompt", type=str, default="")
    parser.add_argument("--neg_prompt", type=str, default="low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--better_start", action="store_true")
    ### common parameters
    parser.add_argument("--seed", type=int, default=231)
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load input image and setup output folder
    lq = np.array(Image.open(args.input).convert("RGB"))
    h, w = lq.shape[:2]
    H, W = int(h * args.upscale), int(w * args.upscale)
    print(f"input image size: {(w, h)}")

    lq_upscaled = bicubic_resize_to(lq, size=(W, H))
    out_folder = Path(args.output)
    out_folder.mkdir(exist_ok=True, parents=True)

    # Initialize stage1 model
    model_stage1: RRDBNet = load_BSRNet(device)
    # Initialize stage2 model
    cldm, diffusion = load_stage2_models(device)

    # Prepare input data
    lq = image_to_tensor(lq, device)
    # Run stage1
    clean = model_stage1(lq)
    print(f"stage1 output size: {(clean.shape[3], clean.shape[2])}")
    # Run stage2
    # utils/helpers:Pipeline:run_stage2
    hq = inference_stage2(cldm, diffusion, clean, args.steps, args.pos_prompt,
                            args.neg_prompt, args.cfg_scale, args.better_start)
    hq = (hq + 1) * 0.5     # [-1, 1] -> [0, 1]
    print(f"stage2 output size: {(hq.shape[3], hq.shape[2])}")

    # Postproces: correct color
    out = wavelet_reconstruction(hq, clean)
    out = tensor_to_image(out)
    out = bicubic_resize_to(out, size=(W, H))
    print(f"output size: {(out.shape[1], out.shape[0])}")

    # Save outputs
    out_stage1 = tensor_to_image(clean)
    out_stage1 = bicubic_resize_to(out_stage1, size=(W, H))
    out_stage2 = tensor_to_image(hq)
    out_stage2 = bicubic_resize_to(out_stage2, size=(W, H))
    vis = np.concatenate([
        np.concatenate([lq_upscaled, out_stage1], 1),
        np.concatenate([out_stage2, out], 1),
    ], 0)
    Image.fromarray(vis).save(out_folder / Path(args.input).name)


MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth"
}

# utils/inference.py:BSRInferenceLoop:init_stage1_model
def load_BSRNet(device) -> RRDBNet:
    model: RRDBNet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
    weights = load_model_from_url(MODELS["bsrnet"])
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    return model

# utils/inference.py:InferenceLoop:init_stage2_model
def load_stage2_models(device) -> Tuple[ControlLDM, Diffusion]:
    cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
    weights = load_model_from_url(MODELS["sd_v21"])
    unused = cldm.load_pretrained_sd(weights)
    print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
    ### load controlnet
    weights_control = load_model_from_url(MODELS["v2"])
    cldm.load_controlnet_from_ckpt(weights_control)
    print(f"strictly load controlnet weight")
    cldm.eval().to(device)
    ### load diffusion
    diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
    diffusion.to(device)
    return cldm, diffusion

# utils/helpers:Pipeline:run_stage2
def inference_stage2(
    cldm: ControlLDM,
    diffusion: Diffusion,
    stage1_output: torch.Tensor,
    steps: int,
    pos_prompt: List[str],
    neg_prompt: List[str],
    cfg_scale: float,
    better_start: bool = False,
    strength: float = 1.0,  # utils.inference.py:InferenceLoop:run
) -> torch.Tensor:

    clean = stage1_output   # align the variable name to the original file
    bs, _, oh, ow = clean.shape
    device = clean.device
    pad_clean = pad_to_multiples_of(clean, multiple=64)
    _, _, h, w = pad_clean.shape

    cond = cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
    uncond = cldm.prepare_condition(pad_clean, [neg_prompt] * bs)

    diffusion_sampler: SpacedSampler = SpacedSampler(diffusion.betas)

    # TODO: set cond_fn

    old_control_scales = cldm.control_scales
    cldm.control_scales = [strength] * 13

    if better_start:
        # use noised low frequency parts from stage1 output as initial noise,
        # which can prevent noise in the background
        _, low_freq = wavelet_decomposition(pad_clean)
        x_0 = cldm.vae_encode(low_freq)
        x_T = diffusion.q_sample(
            x_0,
            torch.full((bs,), diffusion.num_timesteps - 1, dtype=torch.long, device=device),
            torch.randn_like(x_0),
        )
    else:
        x_T = torch.randn((bs, 4, h // 8, w // 8), device=device)

    z = diffusion_sampler.sample(
        model=cldm,
        device=device,
        steps=steps,
        batch_size=bs,
        x_size=(4, h // 8, w // 8),
        cond=cond,
        uncond=uncond,
        cfg_scale=cfg_scale,
        x_T=x_T,
        progress=True,
        progress_leave=True,
        cond_fn=None, # TODO
        tiled=False,
    )

    x = cldm.vae_decode(z)
    x = x[:, :, :oh, :ow]

    cldm.control_scales = old_control_scales
    return x

def bicubic_resize_to(img: np.ndarray, size: Tuple[int, int] = None, scale: float = None) -> np.ndarray:
    pil = Image.fromarray(img)
    if size is None:
        res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    else:
        res = pil.resize(size, Image.BICUBIC)
    return np.array(res)

def image_to_tensor(image: np.ndarray, device) -> torch.Tensor:
    tensor = torch.tensor((image / 255.).clip(0, 1), dtype=torch.float32, device=device)
    tensor = torch.moveaxis(tensor, -1, 0).unsqueeze(0).contiguous()
    return tensor

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy()
    image = np.clip(np.moveaxis(image[0] * 255, 0, -1), 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    return image

if __name__ == "__main__":
    main()
