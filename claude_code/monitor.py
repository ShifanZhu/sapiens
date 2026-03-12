"""Training monitor: GPU stats, NVMe health, and training metrics.

Usage:
    python monitor.py --run-dir runs/exp001 [--interval 5]
"""

import argparse
import csv
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# ANSI colours
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

GPU_WARN_TEMP  = 80   # °C — yellow
GPU_CRIT_TEMP  = 88   # °C — red, shutdown risk
GPU_WARN_POWER = 320  # W  — yellow (approaching 350 W TDP)

# Log file that survives reboots
LOG_FILE: Path | None = None

def log(msg: str) -> None:
    """Print to stdout and append (ANSI-stripped) to log file."""
    print(msg, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a") as f:
            f.write(_ANSI_RE.sub("", msg) + "\n")


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Thread 1: GPU stats (polled every --interval seconds)
# ---------------------------------------------------------------------------
def gpu_monitor(interval: int, stop: threading.Event) -> None:
    query = "temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    while not stop.is_set():
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            temp_s, pwr_s, mem_used_s, mem_tot_s, util_s = [x.strip() for x in out.split(",")]
            temp  = float(temp_s)
            power = float(pwr_s)
            mem   = f"{float(mem_used_s)/1024:.1f}/{float(mem_tot_s)/1024:.1f} GB"

            temp_col  = RED if temp  >= GPU_CRIT_TEMP  else (YELLOW if temp  >= GPU_WARN_TEMP  else GREEN)
            power_col = RED if power >= GPU_WARN_POWER else GREEN

            log(
                f"[{ts()}] GPU  "
                f"temp={temp_col}{temp:.0f}°C{RESET}  "
                f"power={power_col}{power:.0f}W{RESET}  "
                f"mem={mem}  util={util_s.strip()}%"
            )

            if temp >= GPU_CRIT_TEMP:
                log(f"{RED}{BOLD}[{ts()}] !! GPU CRITICAL TEMP {temp}°C — shutdown risk !!{RESET}")
        except Exception as e:
            log(f"[{ts()}] GPU query failed: {e}")

        stop.wait(interval)


# ---------------------------------------------------------------------------
# Thread 2: NVMe health via journalctl -f (real-time, no polling delay)
# ---------------------------------------------------------------------------
def nvme_monitor(stop: threading.Event) -> None:
    cmd = ["journalctl", "-f", "-k", "--no-pager", "-n", "0"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except Exception as e:
        print(f"[{ts()}] Cannot start journalctl: {e}", flush=True)
        return

    try:
        for line in proc.stdout:
            if stop.is_set():
                break
            low = line.lower()
            if "rxerr" in low or "pcie bus error" in low:
                log(f"{RED}{BOLD}[{ts()}] !! NVME PCIe RxErr !! {line.rstrip()}{RESET}")
            elif "hung_task" in low or "kernel panic" in low or "hard lockup" in low:
                log(f"{RED}{BOLD}[{ts()}] !! KERNEL ERROR: {line.rstrip()}{RESET}")
    finally:
        proc.terminate()


# ---------------------------------------------------------------------------
# Thread 3: Training metrics from metrics.csv
# CSV schema: epoch,lr_backbone,lr_head,train_loss,train_mpjpe_body,
#             val_loss,val_mpjpe_all,val_mpjpe_body,val_mpjpe_hand,epoch_time
# ---------------------------------------------------------------------------
def metrics_monitor(run_dir: Path, interval: int, stop: threading.Event) -> None:
    csv_path = run_dir / "metrics.csv"
    print(f"[{ts()}] Watching {csv_path} ...", flush=True)

    while not csv_path.exists() and not stop.is_set():
        stop.wait(interval)

    last_epoch = -1
    while not stop.is_set():
        try:
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                epoch = int(row["epoch"])
                if epoch <= last_epoch:
                    continue
                last_epoch = epoch
                parts = [f"epoch={epoch}"]
                if row.get("train_loss"):
                    parts.append(f"loss={float(row['train_loss']):.4f}")
                if row.get("train_mpjpe_body"):
                    parts.append(f"train_mpjpe={float(row['train_mpjpe_body']):.1f}mm")
                if row.get("val_mpjpe_body"):
                    parts.append(f"val_mpjpe_body={float(row['val_mpjpe_body']):.1f}mm")
                if row.get("val_mpjpe_all"):
                    parts.append(f"val_mpjpe_all={float(row['val_mpjpe_all']):.1f}mm")
                if row.get("epoch_time"):
                    parts.append(f"time={float(row['epoch_time']):.0f}s")
                log(f"{CYAN}[{ts()}] TRAIN  " + "  ".join(parts) + RESET)
        except Exception:
            pass
        stop.wait(interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/exp001")
    parser.add_argument("--interval", type=int, default=5, help="GPU/metrics poll interval in seconds")
    args = parser.parse_args()

    global LOG_FILE
    run_dir = Path(args.run_dir)
    interval = args.interval
    LOG_FILE = run_dir / "monitor.log"

    print(f"{BOLD}=== Training Monitor (interval={interval}s) ==={RESET}")
    print(f"  run dir: {run_dir}")
    print(f"  log file: {LOG_FILE}  ← survives reboots")
    print(f"  GPU warn thresholds: {GPU_WARN_TEMP}°C / {GPU_CRIT_TEMP}°C (crit)  |  {GPU_WARN_POWER}W power")
    print(f"  NVMe: real-time via journalctl (no polling delay)")
    print(flush=True)

    stop = threading.Event()
    threads = [
        threading.Thread(target=gpu_monitor,     args=(interval, stop), daemon=True),
        threading.Thread(target=nvme_monitor,    args=(stop,),          daemon=True),
        threading.Thread(target=metrics_monitor, args=(run_dir, interval, stop), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] Monitor stopped.")
        stop.set()


if __name__ == "__main__":
    main()
