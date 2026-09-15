import importlib
import os
import sys
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# Add project root to path to enable imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import ExperimentLogger


@hydra.main(config_path="../configs", config_name="globals", version_base=None)
def main(cfg: DictConfig):
    # Validate required parameters
    if not hasattr(cfg, "approach") or cfg.approach is None:
        cfg.approach = "toponet"
        # raise ValueError(
        #     "'approach' must be specified either in config or via CLI (approach=<name>)",
        # )
    if not hasattr(cfg, "experiment") or cfg.experiment is None:
        cfg.experiment = "exp_v0"
        # raise ValueError(
        #     "'experiment' must be specified either in config or via CLI (experiment=<name>)",
        # )

    # Load experiment config
    exp_path = Path(f"src/approach/{cfg.approach}/configs/experiments/{cfg.experiment}.yaml")
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    exp_cfg = OmegaConf.load(exp_path)
    OmegaConf.set_struct(cfg, False)
    cfg = DictConfig(OmegaConf.merge(cfg, exp_cfg))

    # Optionally create dataset before training
    if getattr(cfg, "create_dataset", False):
        # Import your dataset creation function here
        # For example: from src.approach.<approach>.dataset import create_dataset
        dataset_module_path = f"src.approach.{cfg.approach}.dataset"
        dataset_module = importlib.import_module(dataset_module_path)
        dataset_module.create_dataset(cfg)
        print("Dataset created. Exiting as requested by create_dataset flag.")
        return

    # Prepare output root and subdirectories
    output_root = Path(cfg.output_dir) / cfg.approach / cfg.experiment
    hydra_dir = output_root / "hydra"
    wandb_dir = output_root / "wandb"
    results_dir = output_root / "results"

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    # hydra_dir will be created by Hydra if you override hydra.run.dir

    # Set WANDB_DIR before wandb.init
    os.environ["WANDB_DIR"] = str(wandb_dir.absolute())
    wandb_run = wandb.init(
        project=cfg.wandb.project_name,
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        name=f"{cfg.approach}_{cfg.experiment}",
        mode=cfg.wandb.mode,
    )

    # Create logger that writes both to wandb + local (results_dir)
    logger = ExperimentLogger(results_dir, run=wandb_run)

    # Add directory paths to config for training script
    OmegaConf.set_struct(conf=cfg, value=False)
    cfg.directories = {
        "results_dir": str(results_dir.absolute()),
        "wandb_dir": str(wandb_dir.absolute()),
        "output_root": str(output_root.absolute()),
    }

    # Load approach and run
    module_path = f"src.approach.{cfg.approach}.train"
    train_module = importlib.import_module(module_path)
    results = train_module.run_experiment(cfg, logger=logger)

    if isinstance(results, dict):
        logger.log_metrics(results)

    wandb_run.finish()

    # NOTE: To have Hydra save its .hydra config/logs in the same output_root, launch your script with:
    # python scripts/run_experiments.py hydra.run.dir=outputs/<approach>/<experiment>/hydra
    # Or set hydra.run.dir in your config using interpolation if you want to automate this.


if __name__ == "__main__":
    main()
