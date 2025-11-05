"""Data preprocessing and cleaning for bankruptcy prediction."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """Preprocess bankruptcy prediction data."""
    
    def __init__(self, config):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Remove duplicate rows
        n_duplicates = df_clean.duplicated().sum()
        if n_duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {n_duplicates} duplicate rows")
            
        # Replace infinite values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many missing values
        missing_threshold = self.config.get('preprocessing.missing_threshold', 0.5)
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
            
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        
        return df_clean
    
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = None,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Handle outliers in numeric features.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold value for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        method = method or self.config.get('preprocessing.outlier_method', 'iqr')
        threshold = threshold or self.config.get('preprocessing.outlier_threshold', 3.0)
        
        logger.info(f"Handling outliers using {method} method...")
        
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_out[col].quantile(0.25)
                Q3 = df_out[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                
        elif method == 'zscore':
            for col in numeric_cols:
                mean = df_out[col].mean()
                std = df_out[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=self.config.random_seed)
            outlier_mask = iso_forest.fit_predict(df_out[numeric_cols]) == -1
            
            # Replace outliers with median
            for col in numeric_cols:
                median_val = df_out[col].median()
                df_out.loc[outlier_mask, col] = median_val
                
        logger.info("Outlier handling completed")
        
        return df_out
    
    def impute_missing_values(
        self, 
        df: pd.DataFrame,
        strategy: str = None
    ) -> pd.DataFrame:
        """
        Impute missing values.
        
        Args:
            df: Input DataFrame with missing values
            strategy: Imputation strategy ('mean', 'median', 'mode', 'knn')
            
        Returns:
            DataFrame with imputed values
        """
        strategy = strategy or self.config.get('preprocessing.imputation_strategy', 'median')
        
        logger.info(f"Imputing missing values using {strategy} strategy...")
        
        df_imputed = df.copy()
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        if strategy in ['mean', 'median', 'most_frequent']:
            self.imputer = SimpleImputer(strategy=strategy)
            df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
            
        elif strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
            
        n_missing_before = df[numeric_cols].isnull().sum().sum()
        n_missing_after = df_imputed[numeric_cols].isnull().sum().sum()
        
        logger.info(f"Imputed {n_missing_before - n_missing_after} missing values")
        
        return df_imputed
    
    def scale_features(
        self, 
        df: pd.DataFrame,
        method: str = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit scaler (True for train, False for test)
            
        Returns:
            DataFrame with scaled features
        """
        method = method or self.config.get('preprocessing.scaling_method', 'robust')
        
        logger.info(f"Scaling features using {method} method...")
        
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
                
            df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            df_scaled[numeric_cols] = self.scaler.transform(df_scaled[numeric_cols])
            
        logger.info("Feature scaling completed")
        
        return df_scaled
    
    def split_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X, y)
        """
        target_col = 'class'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def create_temporal_split(
        self,
        df: pd.DataFrame,
        train_size: float = None,
        val_size: float = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create temporal train/val/test split based on year.
        
        Args:
            df: Input DataFrame with 'year' column
            train_size: Proportion for training
            val_size: Proportion for validation
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        train_size = train_size or self.config.get('data.train_size', 0.7)
        val_size = val_size or self.config.get('data.val_size', 0.15)
        
        logger.info("Creating temporal split...")
        
        if 'year' not in df.columns:
            raise ValueError("DataFrame must contain 'year' column for temporal split")
            
        years = sorted(df['year'].unique())
        n_years = len(years)
        
        train_years = years[:int(n_years * train_size)]
        val_years = years[int(n_years * train_size):int(n_years * (train_size + val_size))]
        test_years = years[int(n_years * (train_size + val_size)):]
        
        train_df = df[df['year'].isin(train_years)].drop(columns=['year'])
        val_df = df[df['year'].isin(val_years)].drop(columns=['year'])
        test_df = df[df['year'].isin(test_years)].drop(columns=['year'])
        
        logger.info(f"Train: {len(train_df)} samples (years {train_years})")
        logger.info(f"Val: {len(val_df)} samples (years {val_years})")
        logger.info(f"Test: {len(test_df)} samples (years {test_years})")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def get_preprocessing_pipeline(self) -> Dict:
        """
        Get fitted preprocessing objects for reproducibility.
        
        Returns:
            Dictionary with scaler and imputer
        """
        return {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns
        }