"""Download and prepare bankruptcy prediction dataset."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config import Config
from src.data.data_loader import BankruptcyDataLoader
from src.utils.helpers import set_seed, timer


@timer
def download_and_prepare_data():
    """Download and prepare dataset."""
    logger.info("Starting data download and preparation...")
    
    # Load configuration
    config = Config()
    set_seed(config.random_seed)
    
    # Initialize data loader
    data_loader = BankruptcyDataLoader(config)
    
    # Download data
    try:
        data_loader.download_data(force=False)
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        logger.info("Attempting to load existing data...")
    
    # Load all years
    df = data_loader.load_all_years()
    
    # Get data info
    info = data_loader.get_data_info()
    logger.info(f"Dataset Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Save combined raw data
    data_loader.save_processed_data(df, 'raw_combined.parquet')
    
    logger.info("Data download and preparation completed!")


if __name__ == "__main__":
    download_and_prepare_data() 