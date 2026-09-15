"""Queue and execute multiple experiments sequentially.

This script discovers all experiment configurations under src/approach/*/configs/experiments/
and runs them one by one using run_experiment.py.

Usage:
    python scripts/queue_experiments.py [approach_filter]

Examples:
    python scripts/queue_experiments.py              # Run all experiments
    python scripts/queue_experiments.py unet         # Run only unet experiments

"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APPROACHES_DIR = BASE_DIR / "src" / "approach"


def main():
    """Discover and run all experiment configurations."""
    # Optional: filter by approach from CLI
    approach_filter = sys.argv[1] if len(sys.argv) > 1 else None

    experiments_found = 0
    experiments_run = 0
    experiments_failed = 0

    # Find all experiment configs under each approach
    for approach_dir in sorted(APPROACHES_DIR.iterdir()):
        if not approach_dir.is_dir():
            continue

        approach_name = approach_dir.name

        # Skip if filtering by approach
        if approach_filter and approach_name != approach_filter:
            continue

        exp_dir = approach_dir / "configs" / "experiments"
        if not exp_dir.exists():
            continue

        for exp_file in sorted(exp_dir.glob("*.yaml")):
            exp_name = exp_file.stem
            experiments_found += 1

            print(f"\n{'=' * 60}")
            print(f"Running experiment {experiments_found}: {approach_name}/{exp_name}")
            print(f"{'=' * 60}")

            cmd = [
                "python",
                str(BASE_DIR / "scripts" / "run_experiment.py"),
                f"approach={approach_name}",
                f"experiment={exp_name}",
            ]

            try:
                subprocess.run(cmd, check=True, cwd=BASE_DIR)
                experiments_run += 1
                print(f"✓ Successfully completed {approach_name}/{exp_name}")
            except subprocess.CalledProcessError as e:
                experiments_failed += 1
                print(f"✗ Failed to run {approach_name}/{exp_name}: {e}")
                print("Continuing with next experiment...")

    # Summary
    print(f"\n{'=' * 60}")
    print("Experiment Queue Summary")
    print(f"{'=' * 60}")
    print(f"Total found:      {experiments_found}")
    print(f"Successfully run: {experiments_run}")
    print(f"Failed:           {experiments_failed}")
    print(f"{'=' * 60}")

    if experiments_found == 0:
        print("No experiments found!")
        if approach_filter:
            print(f"Hint: No experiments found for approach '{approach_filter}'")
        return 1

    return 0 if experiments_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
