import argparse
import os
import subprocess
import sys
import time
import yaml


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _env_for_gpu(gpu):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def main():
    p = argparse.ArgumentParser(description="Launch run_multi.py per-dataset on specific GPUs (parallel) with optional nohup-friendly logging.")
    p.add_argument("--config", required=True, help="Path to multi_ucr-style yaml.")
    p.add_argument("--max_parallel", type=int, default=None, help="Max concurrent datasets. Default: number of datasets")
    p.add_argument("--poll", type=float, default=5.0, help="Polling interval in seconds")
    p.add_argument("--logs_dir", default="./multi_logs", help="Where to write per-dataset logs")
    args = p.parse_args()

    multi_cfg = _load_yaml(args.config)
    datasets = list(multi_cfg.get("datasets", []))
    if not datasets:
        raise SystemExit("No datasets found in config")

    os.makedirs(args.logs_dir, exist_ok=True)

    max_parallel = args.max_parallel or len(datasets)

    pending = datasets[:]
    running = []  # list of tuples (name, proc)

    def start_one(ds):
        name = ds.get("name")
        run_name = ds.get("run_name", f"{name.lower()}_run")
        gpu = ds.get("gpu")
        log_path = os.path.join(args.logs_dir, f"{run_name}.log")
        log_f = open(log_path, "w")
        cmd = [sys.executable, "run_multi.py", "--config", args.config]
        env = _env_for_gpu(gpu)
        env["RCF_SINGLE_DATASET"] = name
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        return run_name, proc, log_f

    while pending or running:
        while pending and len(running) < max_parallel:
            ds = pending.pop(0)
            run_name, proc, log_f = start_one(ds)
            print(f"[run_multi_parallel] started {run_name} pid={proc.pid}")
            running.append((run_name, proc, log_f))

        time.sleep(args.poll)

        still_running = []
        for run_name, proc, log_f in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((run_name, proc, log_f))
            else:
                log_f.close()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[run_multi_parallel] finished {run_name}: {status}")
        running = still_running


if __name__ == "__main__":
    main()

