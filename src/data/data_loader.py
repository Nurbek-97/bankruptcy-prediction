"""Data loading and acquisition for bankruptcy prediction."""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import pandas as pd
from loguru import logger
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class BankruptcyDataLoader:
    """Load and manage bankruptcy dataset."""
    
    def __init__(self, config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.raw_data_path = Path(config.get('data.raw_data_path'))
        self.dataset_url = config.get('data.dataset_url')
        
    def download_data(self, force: bool = False) -> None:
        """
        Download Polish bankruptcy dataset from UCI repository.
        
        Args:
            force: Force re-download if data exists
        """
        zip_path = self.raw_data_path / "data.zip"
        
        if zip_path.exists() and not force:
            logger.info("Dataset already exists. Use force=True to re-download.")
            return
            
        logger.info(f"Downloading dataset from {self.dataset_url}")
        
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
                urlretrieve(
                    self.dataset_url,
                    filename=zip_path,
                    reporthook=t.update_to
                )
            logger.info(f"Dataset downloaded to {zip_path}")
            
            # Extract files
            self._extract_zip(zip_path)
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
            
    def _extract_zip(self, zip_path: Path) -> None:
        """
        Extract zip file contents.
        
        Args:
            zip_path: Path to zip file
        """
        logger.info("Extracting dataset files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_data_path)
            
        logger.info(f"Files extracted to {self.raw_data_path}")
        
    def load_year_data(self, year: int) -> pd.DataFrame:
        """
        Load data for specific year.
        
        Args:
            year: Year number (1-5)
            
        Returns:
            DataFrame with bankruptcy data
        """
        file_path = self.raw_data_path / f"{year}year.arff"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.info(f"Loading data for year {year}")
        
        # Read ARFF file
        df = self._read_arff(file_path)
        df['year'] = year
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def _read_arff(self, file_path: Path) -> pd.DataFrame:
        """
        Read ARFF file format.
        
        Args:
            file_path: Path to ARFF file
            
        Returns:
            DataFrame
        """
        from scipy.io import arff
        
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Decode byte strings
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')
                
        # Convert class column to binary
        if 'class' in df.columns:
            df['class'] = (df['class'] == '1').astype(int)
            
        return df
    
    def load_all_years(self) -> pd.DataFrame:
        """
        Load and combine data from all years.
        
        Returns:
            Combined DataFrame
        """
        logger.info("Loading data from all years...")
        
        dfs = []
        for year in range(1, 6):
            try:
                df = self.load_year_data(year)
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Data for year {year} not found, skipping")
                
        if not dfs:
            raise ValueError("No data files found. Run download_data() first.")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Combined dataset: {len(combined_df)} samples, {len(combined_df.columns)} features")
        logger.info(f"Class distribution:\n{combined_df['class'].value_counts()}")
        
        return combined_df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature column names
        """
        df = self.load_year_data(1)
        feature_cols = [col for col in df.columns if col not in ['class', 'year']]
        return feature_cols
    
    def get_data_info(self) -> Dict:
        """
        Get dataset statistics and information.
        
        Returns:
            Dictionary with dataset info
        """
        df = self.load_all_years()
        
        info = {
            'total_samples': len(df),
            'n_features': len(df.columns) - 2,  # Exclude class and year
            'n_bankruptcy': df['class'].sum(),
            'n_non_bankruptcy': len(df) - df['class'].sum(),
            'bankruptcy_rate': df['class'].mean() * 100,
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'years': sorted(df['year'].unique()),
        }
        
        return info
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to parquet format.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        processed_path = Path(self.config.get('data.processed_data_path'))
        output_path = processed_path / filename
        
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Data saved to {output_path}")
        
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from parquet format.
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame
        """
        processed_path = Path(self.config.get('data.processed_data_path'))
        input_path = processed_path / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found: {input_path}")
            
        df = pd.read_parquet(input_path)
        logger.info(f"Data loaded from {input_path}")
        
        return df