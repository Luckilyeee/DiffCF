import argparse
import os
import subprocess
import sys
import yaml


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _run(script, config_path):
    cmd = [sys.executable, script, "--config", config_path]
    subprocess.run(cmd, check=True)


def _prepare_cfg(cfg, use_cone):
    cfg = dict(cfg)
    sampling = dict(cfg.get("sampling", {}))
    sampling["use_temporal_cone"] = bool(use_cone)
    cfg["sampling"] = sampling
    run_name = cfg.get("run_name", "run")
    if use_cone:
        if not run_name.endswith("_cone"):
            run_name = f"{run_name}_cone"
    else:
        if run_name.endswith("_cone"):
            run_name = run_name[: -len("_cone")]
    cfg["run_name"] = run_name
    return cfg


def main(config_path, use_cone):
    cfg = _prepare_cfg(_load_yaml(config_path), use_cone)
    run_dir = os.path.join(cfg["output"]["root"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    resolved_path = os.path.join(run_dir, "resolved_config.yaml")
    _write_yaml(resolved_path, cfg)

    clf_ckpt = os.path.join(run_dir, "classifier.pt")
    diff_ckpt = os.path.join(run_dir, "diffusion.pt")

    if not os.path.exists(clf_ckpt):
        _run("train_classifier.py", resolved_path)
    if not os.path.exists(diff_ckpt):
        _run("train_diffusion.py", resolved_path)

    _run("run_generate_cf.py", resolved_path)
    _run("run_evaluate.py", resolved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cone", action="store_true", help="Enable temporal cone guidance and use *_cone run_name")
    args = parser.parse_args()
    main(args.config, args.cone)
