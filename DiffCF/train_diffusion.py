import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import load_datasets
from src.models.unet1d_diffusion import UNet1D
from src.diffusion.ddpm import GaussianDiffusion


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, _, meta = load_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"])

    model = UNet1D(meta["channels"], base_channels=cfg["diffusion"]["model_channels"], depth=cfg["diffusion"]["depth"]).to(device)
    diffusion = GaussianDiffusion(cfg["diffusion"]["timesteps"], cfg["diffusion"]["schedule"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["diffusion"]["lr"])
    ema = EMA(model, decay=cfg["diffusion"]["ema_decay"])

    max_t_ratio = float(cfg["diffusion"].get("max_t_ratio", 1.0))
    max_t_ratio = min(max(max_t_ratio, 0.05), 1.0)
    max_t = max(1, int(diffusion.timesteps * max_t_ratio))

    loss_type = cfg["diffusion"].get("loss_type", "mse")
    lambda_tv = float(cfg["diffusion"].get("lambda_tv", 0.01))
    lambda_smooth = float(cfg["diffusion"].get("lambda_smooth", 0.0))

    run_dir = os.path.join(cfg["output"]["root"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "diffusion.pt")

    for epoch in range(cfg["diffusion"]["epochs"]):
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            t = torch.randint(0, max_t, (x.size(0),), device=device)
            loss = diffusion.training_losses(model, x, t, loss_type=loss_type, lambda_tv=lambda_tv, lambda_smooth=lambda_smooth)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            total_loss += loss.item() * x.size(0)
        print(f"[Diffusion] epoch={epoch} loss={total_loss/len(train_ds):.4f} max_t={max_t} loss_type={loss_type}")

    torch.save({"state_dict": model.state_dict(), "ema": ema.shadow, "meta": meta}, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
