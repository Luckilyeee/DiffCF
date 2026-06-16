import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

import argparse
import os
import yaml
import torch
from src.data.dataset import load_datasets
from src.models.unet1d_diffusion import UNet1D
from src.diffusion.ddpm import GaussianDiffusion
from src.cf.diffusion_sanity import run_sanity_from_dataset


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds, _, meta = load_datasets(cfg)
    model = UNet1D(meta["channels"], base_channels=cfg["diffusion"]["model_channels"], depth=cfg["diffusion"]["depth"]).to(device)
    diffusion = GaussianDiffusion(cfg["diffusion"]["timesteps"], cfg["diffusion"]["schedule"])

    run_dir = os.path.join(cfg["output"]["root"], cfg["run_name"])
    ckpt_path = os.path.join(run_dir, "diffusion.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema", ckpt["state_dict"]), strict=True)

    _, plot_path = run_sanity_from_dataset(model, diffusion, train_ds, cfg, args.out_dir, device=device,
                                           batch_size=args.batch_size, seed=args.seed)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

