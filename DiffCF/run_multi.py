import argparse
import os
import subprocess
import sys
import yaml
import numpy as np


STAGE_TO_SCRIPT = {
    "train_classifier": "train_classifier.py",
    "train_diffusion": "train_diffusion.py",
    "generate_cf": "run_generate_cf.py",
    "evaluate": "run_evaluate.py",
}


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _infer_ucr_meta(root, name):
    train_path = os.path.join(root, name, name + "_TRAIN.tsv")
    data = np.loadtxt(train_path, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    length = x.shape[1]
    num_classes = len(np.unique(y))
    return length, num_classes


def _apply_overrides(cfg, dataset_cfg):
    cfg = dict(cfg)
    dataset = dict(cfg.get("dataset", {}))
    dataset["name"] = dataset_cfg["name"]
    if "length" in dataset_cfg:
        dataset["length"] = dataset_cfg["length"]
    if "num_classes" in dataset_cfg:
        dataset["num_classes"] = dataset_cfg["num_classes"]
    if "batch_size" in dataset_cfg:
        dataset["batch_size"] = dataset_cfg["batch_size"]
    if "length" not in dataset_cfg or "num_classes" not in dataset_cfg:
        root = dataset.get("root")
        if root is None:
            raise ValueError("dataset.root is required to infer length/num_classes")
        length, num_classes = _infer_ucr_meta(root, dataset_cfg["name"])
        dataset.setdefault("length", length)
        dataset.setdefault("num_classes", num_classes)
    cfg["dataset"] = dataset
    cfg["run_name"] = dataset_cfg.get("run_name", f"{dataset_cfg['name'].lower()}_run")

    if "gpu" in dataset_cfg:
        runtime = dict(cfg.get("runtime", {}))
        runtime["cuda_visible_devices"] = dataset_cfg["gpu"]
        cfg["runtime"] = runtime

    sampling = dict(cfg.get("sampling", {}))
    if dataset.get("num_classes", 0) > 2:
        sampling.setdefault("target_rule", "next_best")
    cfg["sampling"] = sampling
    return cfg


def _apply_loss_variant(cfg, variant):
    if not variant:
        return cfg
    cfg = dict(cfg)
    diffusion = dict(cfg.get("diffusion", {}))
    if "loss_type" in variant:
        diffusion["loss_type"] = variant["loss_type"]
    if "lambda_tv" in variant:
        diffusion["lambda_tv"] = variant["lambda_tv"]
    if "lambda_smooth" in variant:
        diffusion["lambda_smooth"] = variant["lambda_smooth"]
    cfg["diffusion"] = diffusion

    suffix = variant.get("run_name_suffix") or variant.get("name")
    if suffix:
        cfg["run_name"] = f"{cfg['run_name']}_{suffix}"
    return cfg


def _run_stage(script, config_path, cuda_visible_devices=None):
    cmd = [sys.executable, script, "--config", config_path]
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    subprocess.run(cmd, check=True, env=env)


def _ckpt_paths(run_dir):
    return {
        "train_classifier": os.path.join(run_dir, "classifier.pt"),
        "train_diffusion": os.path.join(run_dir, "diffusion.pt"),
    }


def _maybe_apply_best_params(cfg):

    output_cfg = cfg.get("output", {}) or {}
    if not bool(output_cfg.get("use_best_params", False)):
        return cfg

    load_root = output_cfg.get("load_root", output_cfg.get("root", "./output"))
    best_path = os.path.join(load_root, cfg["run_name"], "best_params.json")
    if not os.path.exists(best_path):

        alt = os.path.join(load_root, cfg["run_name"], "best_w_smooth.json")
        if os.path.exists(alt):
            best_path = alt
        else:
            print(f"[run_multi] use_best_params=true but missing: {best_path}")
            return cfg

    with open(best_path, "r") as f:
        best = yaml.safe_load(f)

    sampling = dict(cfg.get("sampling", {}))

    for k in ["w_cls", "w_dist", "w_smooth", "start_ratio", "step_size"]:
        if k in best and best[k] is not None:
            sampling[k] = float(best[k])
    cfg = dict(cfg)
    cfg["sampling"] = sampling
    cfg.setdefault("meta", {})
    cfg["meta"]["best_params_path"] = best_path
    print(f"[run_multi] Loaded best params from: {best_path} -> {sampling}")
    return cfg


def main(args):
    multi_cfg = _load_yaml(args.config)
    base_cfg = _load_yaml(multi_cfg["base_config"])
    stages = multi_cfg.get("stages", list(STAGE_TO_SCRIPT.keys()))

    datasets = list(multi_cfg["datasets"])
    only = os.environ.get("RCF_SINGLE_DATASET")
    if only:
        datasets = [d for d in datasets if d.get("name") == only]
        if not datasets:
            raise ValueError(f"RCF_SINGLE_DATASET={only} not found in config")

    loss_variants = multi_cfg.get("loss_variants") or [None]

    for dataset_cfg in datasets:
        for variant in loss_variants:
            cfg = _apply_overrides(base_cfg, dataset_cfg)
            cfg = _apply_loss_variant(cfg, variant)
            cfg = _maybe_apply_best_params(cfg)
            out_root = cfg.get("output", {}).get("root", "./outputs")
            run_dir = os.path.join(out_root, cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            resolved_path = os.path.join(run_dir, "resolved_config.yaml")
            _write_yaml(resolved_path, cfg)
            ckpts = _ckpt_paths(run_dir)

            for stage in stages:
                script = STAGE_TO_SCRIPT.get(stage)
                if script is None:
                    raise ValueError(f"Unknown stage: {stage}")
                if stage in ckpts and os.path.exists(ckpts[stage]):
                    print(f"[run_multi] Skip {stage}, checkpoint exists: {ckpts[stage]}")
                    continue
                _run_stage(script, resolved_path, dataset_cfg.get("gpu"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
