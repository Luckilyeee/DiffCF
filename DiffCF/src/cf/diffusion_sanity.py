import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.diffusion.ddim import get_ddim_timesteps, ddim_step


def run_diffusion_only_sanity_check(model, diffusion, x0, cfg, out_dir, seed=0):
    """Run diffusion-only sampling from a noisy initialization of real x0.

    Guidance is disabled by bypassing gradient updates entirely.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = x0.device
    model.eval()

    sampling_cfg = cfg.get("sampling", {})
    ddim_steps = int(cfg["diffusion"].get("ddim_steps", 100))
    eta = float(sampling_cfg.get("eta", 0.0))
    start_ratio = float(sampling_cfg.get("start_ratio", 0.6))

    timesteps = get_ddim_timesteps(ddim_steps, diffusion.timesteps, start_ratio=start_ratio)
    t_start = timesteps[0].item()
    t_batch = torch.full((x0.shape[0],), t_start, device=device, dtype=torch.long)
    x_t = diffusion.q_sample(x0, t_batch)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x0.shape[0],), t.item(), device=device, dtype=torch.long)
            eps_pred = model(x_t, t_batch)
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1
            x_t, x0_hat = ddim_step(x_t, eps_pred, t.item(), t_prev, diffusion.alpha_bar.to(device), eta=eta)

    x_gen = x0_hat.detach().cpu()
    x_orig = x0.detach().cpu()
    os.makedirs(out_dir, exist_ok=True)
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    plot_path = os.path.join(out_dir, f"diffusion_sanity_check_{dataset_name}.png")
    _plot_overlays(x_orig, x_gen, plot_path)
    return x_gen, plot_path


def _plot_overlays(x_orig, x_gen, out_path, n_samples=6):
    x_orig = x_orig.cpu().numpy() if isinstance(x_orig, torch.Tensor) else np.asarray(x_orig)
    x_gen = x_gen.cpu().numpy() if isinstance(x_gen, torch.Tensor) else np.asarray(x_gen)
    n = min(x_orig.shape[0], x_gen.shape[0], n_samples)

    idx = np.random.choice(x_orig.shape[0], size=n, replace=False) if x_orig.shape[0] > n else np.arange(n)
    x_orig = x_orig[idx]
    x_gen = x_gen[idx]

    n_rows = n
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    t = np.arange(x_orig.shape[-1])
    for i in range(n_rows):
        ax = axes[i]
        for c in range(x_orig.shape[1]):
            ax.plot(t, x_orig[i, c], color="tab:blue", linewidth=1.0, alpha=0.8, label="x0" if c == 0 else None)
            ax.plot(t, x_gen[i, c], color="tab:orange", linewidth=1.0, alpha=0.8, label="diffusion-only" if c == 0 else None)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_sanity_from_dataset(model, diffusion, dataset, cfg, out_dir, device="cpu", batch_size=16, seed=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x0, _ = next(iter(loader))
    x0 = x0.to(device)
    return run_diffusion_only_sanity_check(model, diffusion, x0, cfg, out_dir, seed=seed)
