import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def _plot_psd_pair(x_orig, x_cf, out_path, title=None, fs=1.0, nperseg=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x_orig = np.asarray(x_orig)
    x_cf = np.asarray(x_cf)
    channels = x_orig.shape[0]

    for c in range(channels):
        f_orig, pxx_orig = welch(x_orig[c], fs=fs, nperseg=nperseg)
        f_cf, pxx_cf = welch(x_cf[c], fs=fs, nperseg=nperseg)
        label_suffix = f" ch{c}" if channels > 1 else ""
        axes[0].semilogy(f_orig, pxx_orig, linewidth=1.2, label=f"orig{label_suffix}")
        axes[1].semilogy(f_cf, pxx_cf, linewidth=1.2, label=f"cf{label_suffix}")

    axes[0].set_title("Original PSD")
    axes[1].set_title("CF PSD")
    axes[0].set_xlabel("Frequency")
    axes[1].set_xlabel("Frequency")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_overlay(x_orig, x_cf, out_dir, prefix, save_psd=True, psd_fs=1.0, psd_nperseg=None):
    os.makedirs(out_dir, exist_ok=True)
    x_orig = np.asarray(x_orig)
    x_cf = np.asarray(x_cf)
    channels = x_orig.shape[0]
    t = np.arange(x_orig.shape[-1])
    fig, axes = plt.subplots(channels, 1, figsize=(6, 6), sharex=True)
    if channels == 1:
        axes = [axes]
    for c in range(channels):
        axes[c].plot(t, x_orig[c], label="x", linewidth=0.8, alpha=0.9)
        axes[c].plot(t, x_cf[c], label="x_cf", linewidth=0.8, alpha=0.9)
        axes[c].legend(loc="upper right")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_overlay.png"))
    plt.close(fig)

    if save_psd:
        psd_path = os.path.join(out_dir, f"{prefix}_psd.png")
        _plot_psd_pair(x_orig, x_cf, psd_path, fs=psd_fs, nperseg=psd_nperseg)


def plot_all_overlays(x_orig_all, x_cf_all, labels, out_path, max_series=None):
    x_cf_all = np.asarray(x_cf_all)
    labels = np.asarray(labels) if labels is not None else np.zeros((x_cf_all.shape[0],), dtype=int)
    if max_series is not None and max_series > 0:
        x_cf_all = x_cf_all[:max_series]
        labels = labels[:max_series]
    channels = x_cf_all.shape[1]
    t = np.arange(x_cf_all.shape[-1])
    fig, axes = plt.subplots(channels, 1, figsize=(6, 6), sharex=True)
    if channels == 1:
        axes = [axes]
    cmap = plt.get_cmap("Set2")
    unique_labels = sorted(list(set(labels.tolist())))
    for c in range(channels):
        for i in range(x_cf_all.shape[0]):
            color = cmap(int(labels[i]) % cmap.N)
            axes[c].plot(t, x_cf_all[i, c], color=color, alpha=0.6, linewidth=0.7)
        axes[c].set_ylabel(f"Value")
    axes[-1].set_xlabel("Time")
    if unique_labels:
        handles = [plt.Line2D([0], [0], color=cmap(int(lbl) % cmap.N), lw=1.2, label=str(lbl)) for lbl in unique_labels]
        axes[0].legend(handles=handles, title="Class", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
