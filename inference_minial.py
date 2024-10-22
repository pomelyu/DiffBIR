from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from model import ControlLDM, Diffusion, RRDBNet, SwinIR, SCUNet
from utils.common import (instantiate_from_config, wavelet_decomposition,
                          wavelet_reconstruction)
from utils.cond_fn import Guidance, MSEGuidance, WeightedMSEGuidance
from utils.helpers import pad_to_multiples_of
from utils.inference import load_model_from_url
from utils.sampler import SpacedSampler


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["sr", "dn", "fr", "sr-v1", "fr-v1"])
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
    ### guidance parameters
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--g_loss", type=str, default="mse", choices=["mse", "w_mse"])
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_start", type=int, default=1001)
    parser.add_argument("--g_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=1)
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

    lq_upscaled = bicubic_resize_to(lq, size_wh=(W, H))
    out_folder = Path(args.output)
    out_folder.mkdir(exist_ok=True, parents=True)

    # Initialize stage1 and stage2 model
    if args.task == "sr":
        model_stage1: RRDBNet = load_BSR(device)
        cldm, diffusion = load_stage2_models(device, version="v2")
    elif args.task in "fr":
        model_stage1: SwinIR = load_BFR(device)
        cldm, diffusion = load_stage2_models(device, version="v2")
    elif args.task == "dn":
        model_stage1: SCUNet = load_BID(device)
        cldm, diffusion = load_stage2_models(device, version="v2")
    elif args.task == "sr-v1":
        model_stage1: SwinIR = load_BSR_v1(device)
        cldm, diffusion = load_stage2_models(device, version="sr-v1")
    elif args.task == "fr-v1":
        model_stage1: SwinIR = load_BSR_v1(device)
        cldm, diffusion = load_stage2_models(device, version="fr-v1")
    else:
        raise NotImplementedError()
    
    if args.guidance:
        cond_fn = init_cond_fn(args.g_loss, args.g_scale, args.g_start, args.g_stop,
                               args.g_space, args.g_repeat)
    else:
        cond_fn = None
    print("Setup finished")

    # Prepare input data
    lq = image_to_tensor(lq, device)
    # Run stage1
    if args.task == "sr":
        clean = inference_BSR(model_stage1, lq, size_hw=(H, W))
    elif args.task in ["fr", "sr-v1", "fr-v1"]:
        clean = inference_SwinIR(model_stage1, lq, size_hw=(H, W))
    elif args.task == "dn":
        clean = inference_SCUNet(model_stage1, lq, size_hw=(H, W))
    else:
        raise RuntimeError()
    print(f"stage1 output size: {(clean.shape[3], clean.shape[2])}")
    # Run stage2
    # utils/helpers:Pipeline:run_stage2
    hq = inference_stage2(cldm, diffusion, clean, args.steps, args.pos_prompt,
                            args.neg_prompt, args.cfg_scale, args.better_start, cond_fn=cond_fn)
    hq = (hq + 1) * 0.5     # [-1, 1] -> [0, 1]
    print(f"stage2 output size: {(hq.shape[3], hq.shape[2])}")

    # Postproces: correct color
    out = wavelet_reconstruction(hq, clean)
    out = tensor_to_image(out)
    out = bicubic_resize_to(out, size_wh=(W, H))
    print(f"output size: {(out.shape[1], out.shape[0])}")

    # Save outputs
    out_stage1 = tensor_to_image(clean)
    out_stage1 = bicubic_resize_to(out_stage1, size_wh=(W, H))
    out_stage2 = tensor_to_image(hq)
    out_stage2 = bicubic_resize_to(out_stage2, size_wh=(W, H))
    vis = np.concatenate([
        np.concatenate([lq_upscaled, out_stage1], 1),
        np.concatenate([out_stage2, out], 1),
    ], 0)
    Image.fromarray(vis).save(out_folder / Path(args.input).name)


MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth",
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
}

### ==== Load stage1 model =====
# utils/inference.py:BSRInferenceLoop:init_stage1_model
def load_BSR(device) -> RRDBNet:
    model: RRDBNet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
    weights = load_model_from_url(MODELS["bsrnet"])
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    return model

def load_BSR_v1(device) -> RRDBNet:
    model: SwinIR = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
    weights = load_model_from_url(MODELS["swinir_general"])
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    return model

def load_BFR(device) -> SwinIR:
    model: SwinIR = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
    weights = load_model_from_url(MODELS["swinir_face"])
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    return model

def load_BID(device) -> SCUNet:
    model: SCUNet = instantiate_from_config(OmegaConf.load("configs/inference/scunet.yaml"))
    weights = load_model_from_url(MODELS["scunet_psnr"])
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    return model

### ==== Run stage1 model =====
def inference_BSR(bsr_model: RRDBNet, lq: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    clean = bsr_model(lq)
    # TODO: official code make sure the shortest side is at least 512. Why?
    return clean

def inference_SwinIR(swinir_model: SwinIR, lq: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    lq = torch.nn.functional.interpolate(lq, size=size_hw)
    _, _, oh, ow = lq.shape
    pad_lq = pad_to_multiples_of(lq, multiple=64)
    clean = swinir_model(pad_lq)
    clean = clean[:, :, :oh, :ow]
    return clean

def inference_SCUNet(scunet_model: SCUNet, lq: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    lq = torch.nn.functional.interpolate(lq, size=size_hw)
    clean = scunet_model(lq)
    return clean

### ==== Load stage2 model =====
# utils/inference.py:InferenceLoop:init_stage2_model
def load_stage2_models(device, version="v2") -> Tuple[ControlLDM, Diffusion]:
    cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
    weights = load_model_from_url(MODELS["sd_v21"])
    unused = cldm.load_pretrained_sd(weights)
    print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
    ### load controlnet
    if version == "v2":
        weights_controlnet = load_model_from_url(MODELS["v2"])
    elif version == "sr-v1":
        weights_controlnet = load_model_from_url(MODELS["v1_general"])
    elif version == "fr-v1":
        weights_controlnet = load_model_from_url(MODELS["v1_general"])
    else:
        raise ValueError(f"Unknown version: {version}")
    cldm.load_controlnet_from_ckpt(weights_controlnet)
    print(f"strictly load controlnet weight")
    cldm.eval().to(device)
    ### load diffusion
    diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
    diffusion.to(device)
    return cldm, diffusion

def init_cond_fn(g_loss: str, g_scale: float, g_start: int, g_stop: int, g_space: int, g_repeat: int) -> Guidance:
    if g_loss == "mse":
        cond_fn_cls = MSEGuidance
    elif g_loss == "w_mse":
        cond_fn_cls = WeightedMSEGuidance
    else:
        raise ValueError(g_loss)
    cond_fn: Guidance
    cond_fn = cond_fn_cls(
        scale=g_scale,
        t_start=g_start,
        t_stop=g_stop,
        space=g_space,
        repeat=g_repeat,
    )
    return cond_fn

### ==== Run stage2 model =====
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
    cond_fn: Guidance = None,
) -> torch.Tensor:

    clean = stage1_output   # align the variable name to the original file
    bs, _, oh, ow = clean.shape
    device = clean.device
    pad_clean = pad_to_multiples_of(clean, multiple=64)
    _, _, h, w = pad_clean.shape

    cond = cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
    uncond = cldm.prepare_condition(pad_clean, [neg_prompt] * bs)

    if cond_fn is not None:
        if isinstance(cond_fn, WeightedMSEGuidance) and pad_clean.shape[1] != 3:
            raise RuntimeError(f"latent diffusion is not compatible with g_loss=w_mse")
        cond_fn.load_target(pad_clean * 2 - 1)

    old_control_scales = cldm.control_scales
    cldm.control_scales = [strength] * 13

    diffusion_sampler: SpacedSampler = SpacedSampler(diffusion.betas)
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
        cond_fn=cond_fn,
        tiled=False,
    )

    x = cldm.vae_decode(z)
    x = x[:, :, :oh, :ow]

    cldm.control_scales = old_control_scales
    return x

### ==== Utility =====
def bicubic_resize_to(img: np.ndarray, size_wh: Tuple[int, int] = None, scale: float = None) -> np.ndarray:
    pil = Image.fromarray(img)
    if size_wh is None:
        res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    else:
        res = pil.resize(size_wh, Image.BICUBIC)
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
