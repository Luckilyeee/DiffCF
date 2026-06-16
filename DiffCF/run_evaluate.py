import os
import argparse
import yaml
import numpy as np
import torch
from src.data.dataset import load_datasets
from src.models.classifier_fcn import FCNClassifier
from src.eval.evaluate import evaluate_results, load_results
from src.eval.plotting import plot_overlay


def _get_load_root(cfg):
    output_cfg = cfg.get("output", {})
    return output_cfg.get("load_root", output_cfg.get("root", "./output"))


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in rows[0].keys()) + "\n")


def _flatten_config(cfg, prefix=""):
    items = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            items.update(_flatten_config(v, prefix=key + "."))
        else:
            items[key] = v
    return items


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, _, meta = load_datasets(cfg)

    run_dir = os.path.join(cfg["output"]["root"], cfg["run_name"])
    load_dir = os.path.join(_get_load_root(cfg), cfg["run_name"])
    clf_ckpt = _safe_torch_load(os.path.join(load_dir, "classifier.pt"), device)
    classifier = FCNClassifier(meta["channels"], meta["num_classes"]).to(device)
    classifier.load_state_dict(clf_ckpt["state_dict"], strict=True)
    classifier.eval()

    train_raw = train_ds.x.cpu().numpy()
    train_y = train_ds.y.cpu().numpy()

    eval_batch_size = int(cfg.get("evaluation", {}).get(
        "batch_size",
        cfg.get("dataset", {}).get("batch_size", 128)
    ))

    results = load_results(run_dir)
    metrics, per_instance = evaluate_results(results, classifier, train_raw, train_y, batch_size=eval_batch_size)

    out_path = os.path.join(run_dir, "metrics.yaml")
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    save_per_instance = bool(cfg.get("output", {}).get("save_per_instance_csv", False))
    if save_per_instance:
        rows = []
        n = per_instance["l1"].shape[0]
        for i in range(n):
            row = {"index": i}
            for k, v in per_instance.items():
                row[k] = float(v[i]) if hasattr(v, "__len__") else float(v)
            rows.append(row)
        _write_csv(os.path.join(run_dir, "metrics_per_instance.csv"), rows)


    valid_mask = per_instance["validity"].astype(bool)


    summary = {
        "dataset": cfg["dataset"]["name"],
        "run_name": cfg["run_name"],
        "validity": metrics.get("validity"),
        "valid_count": int(np.sum(valid_mask)),
        "total_count": int(valid_mask.shape[0]),
    }

    gen_times = results.get("gen_time_sec")
    if gen_times is not None:
        summary["gen_time_mean_sec"] = float(np.mean(gen_times))
        summary["gen_time_std_sec"] = float(np.std(gen_times))
    total_time = results.get("generation_total_sec")
    if total_time is not None and len(total_time) > 0:
        summary["gen_time_total_sec"] = float(total_time[0])


    metric_keys = [
        "l1",
        "l2",
        "l_inf",
        "rtv",
        "acf1_drop",
        "wasserstein",
        "hf_ratio_delta",
        "target_conf",
    ]

    for key in metric_keys:
        values = per_instance.get(key)
        if values is None:
            continue
        valid_values = values[valid_mask]  # 再次确保统计结果只包含成功样本！
        if valid_values.size == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
        else:
            summary[f"{key}_mean"] = float(np.mean(valid_values))
            summary[f"{key}_std"] = float(np.std(valid_values))

    flat_cfg = _flatten_config({
        "dataset": cfg.get("dataset", {}),
        "classifier": cfg.get("classifier", {}),
        "diffusion": cfg.get("diffusion", {}),
        "sampling": cfg.get("sampling", {}),
        "stabilization": cfg.get("stabilization", {}),
    })
    summary.update(flat_cfg)

    _write_csv(os.path.join(run_dir, "metrics_summary.csv"), [summary])

    print(f"✅ Saved clean metrics to {out_path}")


    eval_cfg = cfg.get("evaluation", {})
    num_plots = int(eval_cfg.get("plot_samples", 0))
    if num_plots > 0:
        plots_dir = os.path.join(run_dir, "plots")
        x_orig = results["x_orig"]
        x_cf = results["x_cf"]
        y_target = results.get("y_target")
        valid_mask = per_instance["validity"].astype(bool)
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            valid_indices = np.arange(x_cf.shape[0])
        rng = np.random.default_rng(int(eval_cfg.get("plot_seed", cfg.get("seed", 0))))
        pick = rng.choice(valid_indices, size=min(num_plots, valid_indices.size), replace=False)
        psd_fs = float(eval_cfg.get("psd_fs", 1.0))
        psd_nperseg = eval_cfg.get("psd_nperseg")
        for idx in pick:
            label = int(y_target[idx]) if y_target is not None else -1
            prefix = f"eval_{idx}_target_{label}"
            plot_overlay(x_orig[idx], x_cf[idx], plots_dir, prefix, save_psd=True, psd_fs=psd_fs, psd_nperseg=psd_nperseg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
