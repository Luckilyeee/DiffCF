import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt


class CFEvaluator:
    """Evaluate plausibility and realism of a single counterfactual time series."""

    def __init__(self, hf_ratio=0.2, psd_fs=1.0, psd_nperseg=None):
        self.hf_ratio = float(hf_ratio)
        self.psd_fs = float(psd_fs)
        self.psd_nperseg = psd_nperseg

    @staticmethod
    def _to_1d(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return x
        if x.ndim == 2:
            return x.reshape(-1)
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")

    @staticmethod
    def _stack_samples(x):
        x = np.asarray(x)
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")

    @staticmethod
    def _total_variation(x):
        return float(np.sum(np.abs(np.diff(x))))

    @staticmethod
    def _acf1(x):
        if x.size < 2:
            return 0.0
        x_centered = x - np.mean(x)
        denom = np.sum(x_centered ** 2) + 1e-8
        return float(np.sum(x_centered[:-1] * x_centered[1:]) / denom)

    def _hf_energy_ratio(self, x):
        x = np.asarray(x)
        fft = np.fft.rfft(x)
        power = (fft.real ** 2 + fft.imag ** 2)
        if power.size == 0:
            return 0.0
        cutoff = max(1, int(np.ceil(power.size * (1.0 - self.hf_ratio))))
        hf_energy = np.sum(power[cutoff:])
        total_energy = np.sum(power) + 1e-8
        return float(hf_energy / total_energy)

    def evaluate(self, x_orig, x_cf, target_class_samples):
        x_orig_1d = self._to_1d(x_orig)
        x_cf_1d = self._to_1d(x_cf)
        target_samples = self._stack_samples(target_class_samples)

        tv_orig = self._total_variation(x_orig_1d)
        tv_cf = self._total_variation(x_cf_1d)
        rtv = float(tv_cf / (tv_orig + 1e-8))

        acf1_cf = self._acf1(x_cf_1d)
        acf1_targets = np.array([self._acf1(self._to_1d(s)) for s in target_samples])
        acf1_target_mean = float(np.mean(acf1_targets))
        acf1_drop = float(acf1_target_mean - acf1_cf)

        target_vals = target_samples.reshape(-1)


        hf_ratio_cf = self._hf_energy_ratio(x_cf_1d)
        hf_ratio_targets = np.array([self._hf_energy_ratio(self._to_1d(s)) for s in target_samples])
        hf_ratio_target_mean = float(np.mean(hf_ratio_targets))
        hf_ratio_delta = float(hf_ratio_cf - hf_ratio_target_mean)

        return {
            "rtv": rtv,
            "acf1_cf": acf1_cf,
            "acf1_target_mean": acf1_target_mean,
            "acf1_drop": acf1_drop,

            "hf_ratio_cf": hf_ratio_cf,
            "hf_ratio_target_mean": hf_ratio_target_mean,
            "hf_ratio_delta": hf_ratio_delta,
        }

    def plot_comparison(self, x_orig, x_cf, save_path, title=None):
        x_orig_1d = self._to_1d(x_orig)
        x_cf_1d = self._to_1d(x_cf)

        f_orig, pxx_orig = welch(x_orig_1d, fs=self.psd_fs, nperseg=self.psd_nperseg)
        f_cf, pxx_cf = welch(x_cf_1d, fs=self.psd_fs, nperseg=self.psd_nperseg)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(x_orig_1d, label="Original", linewidth=1.5)
        axes[0].plot(x_cf_1d, label="CF", linewidth=1.5)
        axes[0].set_title("Time Series")
        axes[0].legend(loc="best")

        axes[1].semilogy(f_orig, pxx_orig, label="Original PSD", linewidth=1.5)
        axes[1].semilogy(f_cf, pxx_cf, label="CF PSD", linewidth=1.5)
        axes[1].set_title("Power Spectral Density")
        axes[1].set_xlabel("Frequency")
        axes[1].legend(loc="best")

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

