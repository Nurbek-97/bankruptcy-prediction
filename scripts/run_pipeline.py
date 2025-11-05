"""Run complete bankruptcy prediction pipeline."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import subprocess
from loguru import logger

from src.config import Config
from src.utils.helpers import timer, print_system_info


@timer
def run_command(command, description):
    """Run a shell command."""
    logger.info("="*50)
    logger.info(description)
    logger.info("="*50)
    
    result = subprocess.run(
        [sys.executable] + command,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        return False
    
    logger.info(f"{description} completed successfully!")
    return True


def main():
    """Run complete pipeline."""
    logger.info("="*60)
    logger.info("BANKRUPTCY PREDICTION - COMPLETE PIPELINE")
    logger.info("="*60)
    
    # Print system info
    print_system_info()
    
    # Load configuration
    config = Config()
    
    scripts_dir = Path(__file__).parent
    
    # Step 1: Download data
    if not run_command(
        [str(scripts_dir / 'download_data.py')],
        "STEP 1: DATA DOWNLOAD"
    ):
        logger.error("Pipeline failed at data download step")
        return
    
    # Step 2: Train models
    if not run_command(
        [str(scripts_dir / 'train.py')],
        "STEP 2: MODEL TRAINING"
    ):
        logger.error("Pipeline failed at training step")
        return
    
    # Step 3: Evaluate best model
    if not run_command(
        [str(scripts_dir / 'evaluate.py'), '--model', 'xgboost'],
        "STEP 3: MODEL EVALUATION"
    ):
        logger.error("Pipeline failed at evaluation step")
        return
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    # Print summary
    logger.info("\nPipeline Summary:")
    logger.info(f"  - Configuration: {config.config_path}")     
    logger.info(f"  - Models saved: {config.get('output.models_path')}")
    logger.info(f"  - Reports: {config.get('output.reports_path')}")
    logger.info(f"  - Figures: {config.get('output.figures_path')}")
    
    logger.info("\nNext Steps:")
    logger.info("  1. Review model performance in reports/")
    logger.info("  2. Check visualizations in reports/figures/")
    logger.info("  3. Use scripts/predict.py for new predictions")
    logger.info("  4. Fine-tune hyperparameters in config/config.yaml")


if __name__ == "__main__":
    main()      