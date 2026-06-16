import os
import numpy as np
import torch

from .metrics import (
    l1_distance_per_instance,
    l2_distance_per_instance,
    l_inf_distance_per_instance,
)
from .cf_evaluator import CFEvaluator


def _predict_probs_batched(classifier, x_cf, device, batch_size):
    probs = []
    n = x_cf.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(x_cf[i:i + batch_size]).float().to(device)
        pb = classifier.predict_proba(xb).detach().cpu()
        probs.append(pb)
    return torch.cat(probs, dim=0)


def evaluate_results(results, classifier, train_raw, train_y, batch_size=128):
    x_orig = results["x_orig"]
    x_cf = results["x_cf"]
    y_target = results["y_target"]

    device = next(classifier.parameters()).device

    with torch.no_grad():
        cf_probs = _predict_probs_batched(classifier, x_cf, device, batch_size)
        cf_pred = cf_probs.argmax(dim=-1).cpu().numpy()
        target_idx = torch.from_numpy(y_target).long()
        target_conf = cf_probs.gather(1, target_idx[:, None]).cpu().numpy().squeeze(-1)


    valid_mask = (cf_pred == y_target)
    validity_rate = float(np.mean(valid_mask))
    has_valid = np.any(valid_mask)

    per_instance = {
        "validity": valid_mask.astype(np.int32),
        "target_conf": target_conf,
        "l1": l1_distance_per_instance(x_orig, x_cf),
        "l2": l2_distance_per_instance(x_orig, x_cf),
        "l_inf": l_inf_distance_per_instance(x_orig, x_cf),
    }

    evaluator = CFEvaluator()
    rtv = np.full((x_cf.shape[0],), np.nan, dtype=np.float32)
    acf1_drop = np.full((x_cf.shape[0],), np.nan, dtype=np.float32)
    hf_ratio_delta = np.full((x_cf.shape[0],), np.nan, dtype=np.float32)

    for i in range(x_cf.shape[0]):
        target_class = int(y_target[i])
        target_samples = train_raw[train_y == target_class]
        if target_samples.size == 0:
            continue
        metrics = evaluator.evaluate(x_orig[i], x_cf[i], target_samples)
        rtv[i] = metrics["rtv"]
        acf1_drop[i] = metrics["acf1_drop"]
        hf_ratio_delta[i] = metrics["hf_ratio_delta"]

    per_instance.update({
        "rtv": rtv,
        "acf1_drop": acf1_drop,
        "hf_ratio_delta": hf_ratio_delta,
    })

    metrics = {
        "validity": validity_rate,
    }

    if has_valid:
        metrics["target_conf"] = float(np.mean(target_conf[valid_mask]))
        metrics["l1"] = float(np.mean(per_instance["l1"][valid_mask]))
        metrics["l2"] = float(np.mean(per_instance["l2"][valid_mask]))
        metrics["l_inf"] = float(np.mean(per_instance["l_inf"][valid_mask]))
        metrics["rtv"] = float(np.nanmean(per_instance["rtv"][valid_mask]))
        metrics["acf1_drop"] = float(np.nanmean(per_instance["acf1_drop"][valid_mask]))
        metrics["hf_ratio_delta"] = float(np.nanmean(per_instance["hf_ratio_delta"][valid_mask]))
    else:
        metrics.update({
            "target_conf": 0.0, "l1": float('inf'), "l2": float('inf'),
            "l_inf": float('inf'), "rtv": float('inf'),
            "acf1_drop": float('inf'),
            "hf_ratio_delta": float('inf')
        })

    return metrics, per_instance


def load_results(run_dir):
    data = np.load(os.path.join(run_dir, "counterfactuals.npz"), allow_pickle=True)
    return data
