import os
import yaml
import copy
import subprocess
import argparse
import json
import pandas as pd
import numpy as np
import time


def _select_best_row(csv_path, validity_threshold):
    if not os.path.exists(csv_path):
        print(f"❌ Missing grid search CSV: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    required_cols = {"Validity", "RTV_Delta", "w_smooth", "w_dist", "L1"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ CSV missing columns {sorted(missing)}: {csv_path}")
        return None

    df = df.copy()
    df["RTV_Delta"] = pd.to_numeric(df["RTV_Delta"], errors="coerce")
    df["Validity"] = pd.to_numeric(df["Validity"], errors="coerce")
    df["L1"] = pd.to_numeric(df["L1"], errors="coerce")
    df = df.dropna(subset=["RTV_Delta", "Validity", "L1"])

    valid_df = df[df["Validity"] >= validity_threshold]
    used_validity = True
    if valid_df.empty:
        valid_df = df
        used_validity = False

    valid_df = valid_df.assign(rtv_abs=valid_df["RTV_Delta"].abs())
    best_row = valid_df.sort_values(["Validity", "rtv_abs", "L1"], ascending=[False, True, True]).iloc[0]
    best_row = best_row.to_dict()
    best_row["validity_passed"] = used_validity
    best_row["validity_threshold"] = float(validity_threshold)
    return best_row


def _build_jobs(multi_cfg, base_cfg):
    datasets = multi_cfg.get("datasets", [])
    loss_variants = multi_cfg.get("loss_variants", [])
    jobs = []

    for ds in datasets:
        ds_name = ds["name"]
        for variant in loss_variants:
            var_name = variant["name"]

            cfg = copy.deepcopy(base_cfg)
            cfg["dataset"]["name"] = ds_name
            if "length" in ds:
                cfg["dataset"]["length"] = ds["length"]
            if "num_classes" in ds:
                cfg["dataset"]["num_classes"] = ds["num_classes"]
            if "batch_size" in ds:
                cfg["dataset"]["batch_size"] = ds["batch_size"]

            cfg["diffusion"]["loss_type"] = variant.get("loss_type", cfg["diffusion"]["loss_type"])
            cfg["diffusion"]["lambda_tv"] = variant.get("lambda_tv", cfg["diffusion"]["lambda_tv"])
            cfg["diffusion"]["lambda_smooth"] = variant.get("lambda_smooth", cfg["diffusion"]["lambda_smooth"])

            run_name_base = ds.get("run_name", f"{ds_name.lower()}_run")
            run_name_suffix = variant.get("run_name_suffix", var_name)
            final_run_name = f"{run_name_base}_{run_name_suffix}"
            cfg["run_name"] = final_run_name

            jobs.append({
                "dataset": ds_name,
                "loss_variant": var_name,
                "run_name": final_run_name,
                "config": cfg,
            })
    return jobs


def _parse_gpu_list(gpu_list):
    if isinstance(gpu_list, list):
        return [str(g) for g in gpu_list]
    gpu_list = str(gpu_list)
    parts = [p.strip() for p in gpu_list.split(",") if p.strip()]
    return parts


def run_multi_grid_search(multi_yaml_path, num_samples, validity_threshold, plot_after, gpu_list, max_parallel):
    with open(multi_yaml_path, "r") as f:
        multi_cfg = yaml.safe_load(f)

    base_config_path = multi_cfg.get("base_config", "configs/coffee.yaml")
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    temp_dir = "configs/temp_grid_search"
    os.makedirs(temp_dir, exist_ok=True)
    logs_dir = "grid_log"
    os.makedirs(logs_dir, exist_ok=True)

    jobs = _build_jobs(multi_cfg, base_cfg)
    summary_rows = []

    gpu_pool = _parse_gpu_list(gpu_list)
    if not gpu_pool:
        raise ValueError("gpu_list is empty. Provide a comma-separated list like '0,1,2,3'.")

    max_parallel = int(max_parallel) if max_parallel else len(gpu_pool)
    max_parallel = max(1, min(max_parallel, len(gpu_pool)))

    print(f"Using GPU pool: {gpu_pool} | max_parallel={max_parallel}")

    available_gpus = gpu_pool.copy()
    running = []

    def _write_temp_yaml(job):
        cfg = job["config"]
        final_run_name = job["run_name"]
        temp_yaml_path = os.path.join(temp_dir, f"{final_run_name}.yaml")
        with open(temp_yaml_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        return temp_yaml_path

    def _get_run_paths(job):
        cfg = job["config"]
        run_dir = os.path.join(cfg["output"]["root"], job["run_name"])
        csv_path = os.path.join(run_dir, "grid_search_results.csv")
        return run_dir, csv_path

    def _record_best(job, csv_path, gpu_id=""):
        run_dir, _ = _get_run_paths(job)
        best_row = _select_best_row(csv_path, validity_threshold)
        if best_row is None:
            return
        best_row.update({
            "dataset": job["dataset"],
            "loss_variant": job["loss_variant"],
            "run_name": job["run_name"],
            "gpu": gpu_id,
        })
        summary_rows.append(best_row)
        best_path = os.path.join(run_dir, "best_params.json")
        with open(best_path, "w") as f:
            json.dump(best_row, f, indent=2)
        print(f"✅ Best config saved: {best_path}")

    def _launch_job(job, gpu_id):
        cfg = job["config"]
        final_run_name = job["run_name"]
        dataset_name = job["dataset"]
        loss_variant = job["loss_variant"]

        temp_yaml_path = _write_temp_yaml(job)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            "python", "grid_search.py",
            "--config", temp_yaml_path,
            "--num_samples", str(num_samples)
        ]

        log_path = os.path.join(logs_dir, f"{dataset_name}_{loss_variant}.log")
        log_f = open(log_path, "a", buffering=1)
        print(f"🚀 Launch: {final_run_name} on GPU {gpu_id} | log={log_path}")
        process = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
        running.append({
            "process": process,
            "gpu": gpu_id,
            "job": job,
            "temp_yaml_path": temp_yaml_path,
            "log_f": log_f,
        })

    def _finalize_job(item):
        job = item["job"]
        gpu_id = item["gpu"]

        run_dir, csv_path = _get_run_paths(job)
        _record_best(job, csv_path, gpu_id)

        if plot_after:
            plot_cmd = ["python", "plot_only.py", "--config", item["temp_yaml_path"]]
            print(f"Plotting: {' '.join(plot_cmd)}")
            subprocess.run(plot_cmd, env=os.environ.copy(), check=False)

    job_queue = []
    for job in jobs:
        run_dir, csv_path = _get_run_paths(job)
        if os.path.exists(csv_path):
            print(f"⏭️  Skip grid search (CSV exists): {csv_path}")
            temp_yaml_path = _write_temp_yaml(job)
            _record_best(job, csv_path, gpu_id="existing")
            if plot_after:
                plot_cmd = ["python", "plot_only.py", "--config", temp_yaml_path]
                print(f"Plotting: {' '.join(plot_cmd)}")
                subprocess.run(plot_cmd, env=os.environ.copy(), check=False)
        else:
            job_queue.append(job)

    while job_queue or running:
        # Launch up to max_parallel
        while job_queue and available_gpus and len(running) < max_parallel:
            job = job_queue.pop(0)
            gpu_id = available_gpus.pop(0)
            _launch_job(job, gpu_id)

        # Poll running processes
        still_running = []
        for item in running:
            ret = item["process"].poll()
            if ret is None:
                still_running.append(item)
                continue
            if ret == 0:
                _finalize_job(item)
                print(f"✅ Finished {item['job']['run_name']}")
            else:
                print(f"❌ Error on {item['job']['run_name']} (exit {ret}), skipping...")
            item["log_f"].close()
            available_gpus.append(item["gpu"])
        running = still_running
        time.sleep(1)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(temp_dir, "best_params_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Best summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_config", required=True, help="Path to your multi_ucr.yaml")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to grid search")
    parser.add_argument("--validity_threshold", type=float, default=0.5, help="Validity cutoff for best selection")
    parser.add_argument("--plot_after", action="store_true", help="Re-plot after each grid search")
    parser.add_argument("--gpu_list", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs to use")
    parser.add_argument("--max_parallel", type=int, default=8, help="Max concurrent jobs")
    args = parser.parse_args()

    run_multi_grid_search(
        args.multi_config,
        args.num_samples,
        args.validity_threshold,
        args.plot_after,
        args.gpu_list,
        args.max_parallel,
    )
