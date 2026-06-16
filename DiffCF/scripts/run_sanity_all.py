
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import argparse
import copy
import os
import subprocess
import tempfile
import time

import numpy as np
import yaml


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
    return x.shape[1], len(np.unique(y))


def _apply_overrides(base_cfg, dataset_cfg):
    cfg = copy.deepcopy(base_cfg)
    dataset = dict(cfg.get("dataset", {}))
    dataset["name"] = dataset_cfg["name"]
    for key in ("length", "num_classes", "batch_size"):
        if key in dataset_cfg:
            dataset[key] = dataset_cfg[key]
    if "length" not in dataset or "num_classes" not in dataset:
        root = dataset.get("root")
        if root is None:
            raise ValueError("dataset.root is required to infer length/num_classes")
        length, num_classes = _infer_ucr_meta(root, dataset_cfg["name"])
        dataset.setdefault("length", length)
        dataset.setdefault("num_classes", num_classes)
    cfg["dataset"] = dataset
    cfg["run_name"] = dataset_cfg.get("run_name", f"{dataset_cfg['name'].lower()}_run")
    return cfg


def _apply_loss_variant(cfg, variant):
    if not variant:
        return cfg
    cfg = copy.deepcopy(cfg)
    diffusion = dict(cfg.get("diffusion", {}))
    for key in ("loss_type", "lambda_tv", "lambda_smooth"):
        if key in variant:
            diffusion[key] = variant[key]
    cfg["diffusion"] = diffusion
    suffix = variant.get("run_name_suffix") or variant.get("name")
    if suffix:
        cfg["run_name"] = f"{cfg['run_name']}_{suffix}"
    return cfg



def main():
    p = argparse.ArgumentParser(description="Run diffusion sanity check for all datasets in a multi_ucr YAML.")
    p.add_argument("--config", required=True, help="Path to multi_ucr-style yaml")
    p.add_argument("--out_dir", default="output/diffusion_sanity",
                   help="Root output dir; a sub-folder per run_name is created")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--poll", type=float, default=5.0, help="Polling interval (seconds)")
    p.add_argument("--max_parallel", type=int, default=None,
                   help="Max concurrent jobs (default: all at once)")
    p.add_argument("--logs_dir", default="./sanity_logs",
                   help="Where to write per-run log files")
    args = p.parse_args()

    multi_cfg = _load_yaml(args.config)
    base_cfg = _load_yaml(multi_cfg["base_config"])
    datasets = list(multi_cfg.get("datasets", []))
    loss_variants = multi_cfg.get("loss_variants") or [None]

    if not datasets:
        raise SystemExit("No datasets found in config")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # Build a job list: one entry per (dataset x loss_variant)
    jobs = []
    tmp_dir = tempfile.mkdtemp(prefix="rcf_sanity_")

    for ds_cfg in datasets:
        for variant in loss_variants:
            cfg = _apply_overrides(base_cfg, ds_cfg)
            cfg = _apply_loss_variant(cfg, variant)

            run_name = cfg["run_name"]
            dataset_name = ds_cfg["name"]
            gpu = ds_cfg.get("gpu")

            # Sub-folder inside out_dir named after the run (e.g. coffee_run_mse)
            job_out_dir = os.path.join(args.out_dir, run_name)
            os.makedirs(job_out_dir, exist_ok=True)

            # Write resolved config to a temp file
            tmp_cfg_path = os.path.join(tmp_dir, f"{run_name}_resolved.yaml")
            _write_yaml(tmp_cfg_path, cfg)

            jobs.append({
                "run_name": run_name,
                "dataset_name": dataset_name,
                "gpu": gpu,
                "cfg_path": tmp_cfg_path,
                "out_dir": job_out_dir,
            })

    max_parallel = args.max_parallel or len(jobs)

    pending = list(jobs)
    running = []   # list of dicts {run_name, proc, log_f}

    sanity_script = str(repo_root / "scripts" / "run_diffusion_sanity_check.py")

    def start_one(job):
        env = os.environ.copy()
        if job["gpu"] is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        device = "cuda:0" if job["gpu"] is not None else "cpu"

        cmd = [
            sys.executable, sanity_script,
            "--config", job["cfg_path"],
            "--out_dir", job["out_dir"],
            "--device", device,
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
        ]
        log_path = os.path.join(args.logs_dir, f"{job['run_name']}_sanity.log")
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        print(f"[sanity_all] started  {job['run_name']}  gpu={job['gpu']}  pid={proc.pid}  log={log_path}")
        return proc, log_f

    while pending or running:
        # fill up to max_parallel
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            proc, log_f = start_one(job)
            running.append({"run_name": job["run_name"], "proc": proc, "log_f": log_f})

        time.sleep(args.poll)

        still_running = []
        for entry in running:
            ret = entry["proc"].poll()
            if ret is None:
                still_running.append(entry)
            else:
                entry["log_f"].close()
                status = "OK" if ret == 0 else f"FAIL(exit={ret})"
                print(f"[sanity_all] finished {entry['run_name']}: {status}")
        running = still_running

    print(f"\n[sanity_all] All done. Plots saved under: {args.out_dir}")


if __name__ == "__main__":
    main()

