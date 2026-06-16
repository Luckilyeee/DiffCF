import os
import argparse
import yaml
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from src.data.dataset import load_datasets
from src.models.classifier_fcn import FCNClassifier
from src.models.unet1d_diffusion import UNet1D
from src.diffusion.ddpm import GaussianDiffusion
from src.cf.generate_cf import generate_counterfactual
from src.eval.plotting import plot_all_overlays


def _get_load_root(cfg):
    output_cfg = cfg.get("output", {})
    return output_cfg.get("load_root", output_cfg.get("root", "./output"))


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _write_csv(path, rows):
    header = [
        "index",
        "y_target",
        "y_pred",
        "y_cf_pred",
        "valid",
        "l1",
        "l2",
        "gen_time_sec",
    ]
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds, meta = load_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    run_dir = os.path.join(cfg["output"]["root"], cfg["run_name"])
    load_dir = os.path.join(_get_load_root(cfg), cfg["run_name"])
    clf_ckpt = _safe_torch_load(os.path.join(load_dir, "classifier.pt"), device)
    diff_ckpt = _safe_torch_load(os.path.join(load_dir, "diffusion.pt"), device)

    classifier = FCNClassifier(meta["channels"], meta["num_classes"]).to(device)
    classifier.load_state_dict(clf_ckpt["state_dict"], strict=True)

    model = UNet1D(meta["channels"], base_channels=cfg["diffusion"]["model_channels"], depth=cfg["diffusion"]["depth"]).to(device)
    model.load_state_dict(diff_ckpt.get("ema", diff_ckpt["state_dict"]), strict=True)
    diffusion = GaussianDiffusion(cfg["diffusion"]["timesteps"], cfg["diffusion"]["schedule"])

    results = {
        "x_orig": [],
        "x_cf": [],
        "y_target": [],
        "y_pred": [],
        "y_cf_pred": [],
        "valid_mask": [],
        "l1": [],
        "l2": [],
        "gen_time_sec": [],
    }

    rows = []
    save_csv = cfg.get("output", {}).get("save_csv", True)
    save_plots = cfg.get("output", {}).get("save_plots", False)
    plot_max = int(cfg.get("output", {}).get("plot_max", 0))
    plot_dir = os.path.join(run_dir, "plots")

    total_start = time.time()

    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        with torch.no_grad():
            y_pred = classifier.predict_proba(x).argmax(dim=-1)
        start_time = time.time()
        x_cf, y_target = generate_counterfactual(model, diffusion, classifier, x, cfg)
        gen_time_sec = time.time() - start_time
        with torch.no_grad():
            y_cf_pred = classifier.predict_proba(x_cf).argmax(dim=-1)
        valid = (y_cf_pred == y_target).item()
        if not valid:
            x_cf = torch.full_like(x_cf, -1.0)
        l1 = torch.mean(torch.abs(x_cf - x)).item() if valid else float("nan")
        l2 = torch.sqrt(torch.mean((x_cf - x) ** 2)).item() if valid else float("nan")

        results["x_orig"].append(x.cpu().numpy())
        results["x_cf"].append(x_cf.cpu().numpy())
        results["y_target"].append(y_target.cpu().numpy())
        results["y_pred"].append(y_pred.cpu().numpy())
        results["y_cf_pred"].append(y_cf_pred.cpu().numpy())
        results["valid_mask"].append(np.array([valid], dtype=np.int32))
        results["l1"].append(l1)
        results["l2"].append(l2)
        results["gen_time_sec"].append(gen_time_sec)

        rows.append([idx, int(y_target.item()), int(y_pred.item()), int(y_cf_pred.item()), int(valid), l1, l2, gen_time_sec])

    total_elapsed = time.time() - total_start

    for k in ["x_orig", "x_cf", "y_target", "y_pred", "y_cf_pred", "valid_mask"]:
        results[k] = np.concatenate(results[k], axis=0)
    results["gen_time_sec"] = np.array(results["gen_time_sec"], dtype=np.float64)
    results["generation_total_sec"] = np.array([total_elapsed], dtype=np.float64)

    np.savez(os.path.join(run_dir, "counterfactuals.npz"), **results)

    if save_csv:
        _write_csv(os.path.join(run_dir, "counterfactuals.csv"), rows)

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, "all_cfs.png")
        plot_all_overlays(results["x_orig"], results["x_cf"], results["y_target"], out_path, max_series=plot_max)

    print(f"Saved counterfactuals to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
