import os
import yaml
import copy
import itertools
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects

from src.data.dataset import load_datasets
from src.models.classifier_fcn import FCNClassifier
from src.models.unet1d_diffusion import UNet1D
from src.diffusion.ddpm import GaussianDiffusion
from src.cf.generate_cf import generate_counterfactual
from src.eval.evaluate import evaluate_results


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_grid_search(config_path, num_samples=20):
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds, meta = load_datasets(base_cfg)


    subset_indices = list(range(min(num_samples, len(test_ds))))
    test_subset = torch.utils.data.Subset(test_ds, subset_indices)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    train_raw = train_ds.x.cpu().numpy()
    train_y = train_ds.y.cpu().numpy()


    run_dir = os.path.join(base_cfg["output"]["root"], base_cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)


    clf_ckpt = _safe_torch_load(os.path.join(run_dir, "classifier.pt"), device)
    classifier = FCNClassifier(meta["channels"], meta["num_classes"]).to(device)
    classifier.load_state_dict(clf_ckpt["state_dict"], strict=True)
    classifier.eval()


    diff_ckpt = _safe_torch_load(os.path.join(run_dir, "diffusion.pt"), device)
    model = UNet1D(meta["channels"], base_cfg["diffusion"]["model_channels"],
                   depth=base_cfg["diffusion"]["depth"]).to(device)
    model.load_state_dict(diff_ckpt.get("ema", diff_ckpt["state_dict"]), strict=True)
    model.eval()
    diffusion = GaussianDiffusion(base_cfg["diffusion"]["timesteps"], base_cfg["diffusion"]["schedule"])


    fixed_params = {
        "start_ratio": 0.2,
        "w_cls": 1.0,
    }


    search_space = {
        "w_dist": [0.0, 1.0, 2.0, 3.0, 4.0],
        "w_smooth": [0.0, 1.0, 2.0, 3.0, 4.0],
    }

    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    summary_results = []

    print(f"Starting Grid Search: {len(combinations)} combinations to test...")
    print(
        f"Fixed Anchors -> start_ratio: {fixed_params['start_ratio']}, w_cls: {fixed_params['w_cls']}\n"
    )

    for idx, params in enumerate(combinations):
        print(
            f"--- [{idx + 1}/{len(combinations)}] Testing -> w_dist: {params['w_dist']}, w_smooth: {params['w_smooth']} ---"
        )


        cfg = copy.deepcopy(base_cfg)


        for k, v in fixed_params.items():
            cfg["sampling"][k] = v


        for k, v in params.items():
            cfg["sampling"][k] = v


        results = {"x_orig": [], "x_cf": [], "y_target": []}

        for x, y in test_loader:
            x = x.to(device)
            x_cf, y_target = generate_counterfactual(model, diffusion, classifier, x, cfg)

            results["x_orig"].append(x.cpu().numpy())
            results["x_cf"].append(x_cf.cpu().numpy())
            results["y_target"].append(y_target.cpu().numpy())


        for k in results.keys():
            results[k] = np.concatenate(results[k], axis=0)


        metrics, _ = evaluate_results(results, classifier, train_raw, train_y)


        row = copy.deepcopy(params)
        row["Validity"] = metrics.get("validity", 0)
        row["L1"] = metrics.get("l1", float('inf'))

        rtv = metrics.get("rtv", float('inf'))
        row["RTV_Delta"] = rtv - 1.0 if np.isfinite(rtv) else float('inf')
        row["HF_Ratio_Delta"] = metrics.get("hf_ratio_delta", float('inf'))

        summary_results.append(row)
        print(
            f"Result -> Validity: {row['Validity']:.2f} | L1: {row['L1']:.3f} | RTV_Delta: {row['RTV_Delta']:.6f}"
            f" | HF_Delta: {row['HF_Ratio_Delta']:.6f}\n")

    # 4. 保存为 CSV
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(run_dir, "grid_search_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Grid search complete! Results saved to {csv_path}")


    plot_pareto(df, run_dir, fixed_params)


def plot_pareto(df, run_dir, fixed_params):
    plt.figure(figsize=(10, 7))


    valid_df = df[df['Validity'] >= 0.3].copy()
    if valid_df.empty:
        print("Warning: No results with Validity >= 0.3 to plot.")
        return
    if "w_dist" not in valid_df.columns:
        raise ValueError("plot_pareto expects column 'w_dist' in grid_search_results.csv")
    norm = mcolors.Normalize(vmin=valid_df['w_dist'].min(), vmax=valid_df['w_dist'].max())
    cmap = cm.plasma


    valid_df = valid_df.assign(rtv_abs=valid_df['RTV_Delta'].abs())


    smooth_levels = sorted(valid_df["w_smooth"].dropna().unique().tolist())

    level_to_base = {}
    base_sizes = [250, 550, 950, 1500, 2200]
    for i, s in enumerate(smooth_levels):
        level_to_base[float(s)] = base_sizes[min(i, len(base_sizes) - 1)]

    base = valid_df["w_smooth"].astype(float).map(level_to_base).fillna(base_sizes[0]).to_numpy()


    inv_rtv = 1.0 / (valid_df['rtv_abs'] + 1e-6)
    if np.isfinite(inv_rtv).all() and inv_rtv.max() > inv_rtv.min():
        q = (inv_rtv - inv_rtv.min()) / (inv_rtv.max() - inv_rtv.min())
        rtv_factor = 0.6 + 0.8 * q  # [0.6, 1.4]
        rtv_factor = rtv_factor.to_numpy() if hasattr(rtv_factor, "to_numpy") else np.asarray(rtv_factor)
    else:
        rtv_factor = np.ones_like(base) * 1.0

    sizes = base * rtv_factor


    scatter = plt.scatter(
        valid_df['L1'],
        valid_df['Validity'],
        s=sizes,
        c=valid_df['w_dist'],
        cmap=cmap,
        norm=norm,
        alpha=0.75,
        marker='o',
        edgecolors="white",
        linewidth=1.5
    )

    plt.colorbar(scatter, ax=plt.gca(), label='w_dist (Proximity Penalty)')


    legend_handles = []
    for s in smooth_levels[:5]:
        handle = plt.scatter([], [], s=level_to_base[float(s)], c="gray", alpha=0.6,
                             edgecolors="white", linewidth=1.0)
        legend_handles.append(handle)
    if legend_handles:
        legend_labels = [f"w_smooth={int(s) if float(s).is_integer() else s}" for s in smooth_levels[:5]]
        plt.legend(legend_handles, legend_labels, title="Bubble base size",
                   loc="lower right", frameon=True)


    ranked = valid_df.sort_values(['Validity', 'rtv_abs', 'L1'], ascending=[False, True, True])
    best_row = ranked.iloc[0]


    plt.scatter(
        [best_row['L1']],
        [best_row['Validity']],
        s=float(np.max(sizes)) * 1.25,
        facecolors='none',
        edgecolors='black',
        linewidth=2.5,
        marker='o',
        zorder=5,
    )

    txt = plt.annotate(
        f"s={best_row['w_smooth']}",
        (best_row['L1'], best_row['Validity']),
        fontsize=11,
        fontweight='bold',
        color='black',
        xytext=(0, 0),
        textcoords='offset points',
        ha='center',
        va='center'
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])


    plt.xlabel('Proximity (L1 Distance) ↓ Better', fontweight='bold')
    plt.ylabel('Validity (%) ↑ Better', fontweight='bold')
    plt.title(
        f"Pareto Frontier (Fixed: start_ratio={fixed_params.get('start_ratio', 'NA')}, w_cls={fixed_params.get('w_cls', 'NA')})\n"
        "Color ~ w_dist | Bubble base size ~ w_smooth | Bubble scale ~ RTV quality (larger = smaller |RTV_Delta|)",
        fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(run_dir, "pareto_frontier_final.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Final high-contrast Pareto plot saved to {plot_path}")



def plot_only_main(config_path):
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    run_dir = os.path.join(base_cfg["output"]["root"], base_cfg["run_name"])
    csv_path = os.path.join(run_dir, "grid_search_results.csv")

    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    print(f"read data: {csv_path}")
    df = pd.read_csv(csv_path)

    fixed_params = {
        "start_ratio": 0.2,
        "w_cls": 1.0,
    }

    plot_pareto(df, run_dir, fixed_params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing CSV")
    args = parser.parse_args()

    if args.plot_only:
        plot_only_main(args.config)
    else:
        run_grid_search(args.config, args.num_samples)
