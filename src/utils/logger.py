import csv
import json
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, output_dir, run=None, filename="metrics_log.csv"):
        """output_dir: path to save local logs (e.g. outputs/decoder_finetune/exp_decoder_clean)
        run: optional wandb run object
        filename: local log file name
        """
        self.run = run
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / filename
        self.jsonl_path = self.output_dir / "metrics_log.jsonl"

        # Initialize CSV with header if new
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "step", "metric", "value"])

    def log_metrics(self, metrics: dict, step=None, console=True):
        """Logs metrics to W&B, CSV, and JSONL file."""
        timestamp = datetime.now().isoformat(timespec="seconds")

        # 1️⃣ W&B Logging
        if self.run is not None:
            self.run.log(metrics, step=step)

        # 2️⃣ Local CSV Logging
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                writer.writerow([timestamp, step, k, v])

        # 3️⃣ Local JSONL Logging
        with open(self.jsonl_path, "a") as f:
            json.dump({"timestamp": timestamp, "step": step, **metrics}, f)
            f.write("\n")

        # 4️⃣ Optional Console Print
        if console:
            metrics_str = ", ".join(
                [
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in metrics.items()
                ]
            )
            print(f"[{timestamp}] step={step} | {metrics_str}")
