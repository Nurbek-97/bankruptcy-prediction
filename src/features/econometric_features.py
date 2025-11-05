"""Econometric feature engineering for bankruptcy prediction."""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class EconometricFeatures:
    """Create econometric features for panel data analysis."""
    
    def __init__(self, config):
        """
        Initialize econometric feature engineer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def create_lag_features(
        self, 
        df: pd.DataFrame,
        lags: list = None,
        group_col: str = None
    ) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.
        
        Args:
            df: DataFrame with time series data
            lags: List of lag periods
            group_col: Column to group by (e.g., company_id)
            
        Returns:
            DataFrame with lag features
        """
        lags = lags or self.config.get('features.lag_features', [1, 2, 3])
        
        logger.info(f"Creating lag features for lags: {lags}")
        
        df_lag = df.copy()
        numeric_cols = df_lag.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        for col in numeric_cols[:10]:  # Limit to avoid too many features
            for lag in lags:
                if group_col:
                    df_lag[f'{col}_lag{lag}'] = df_lag.groupby(group_col)[col].shift(lag)
                else:
                    df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag)
                    
        logger.info(f"Created {len(lags) * min(10, len(numeric_cols))} lag features")
        
        return df_lag
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: list = None,
        group_col: str = None
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: DataFrame with time series data
            windows: List of window sizes
            group_col: Column to group by
            
        Returns:
            DataFrame with rolling features
        """
        windows = windows or self.config.get('features.rolling_windows', [3, 6, 12])
        
        logger.info(f"Creating rolling features for windows: {windows}")
        
        df_roll = df.copy()
        numeric_cols = df_roll.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        for col in numeric_cols[:5]:  # Limit features
            for window in windows:
                if group_col:
                    grouped = df_roll.groupby(group_col)[col]
                    df_roll[f'{col}_roll_mean_{window}'] = grouped.rolling(window).mean().reset_index(0, drop=True)
                    df_roll[f'{col}_roll_std_{window}'] = grouped.rolling(window).std().reset_index(0, drop=True)
                else:
                    df_roll[f'{col}_roll_mean_{window}'] = df_roll[col].rolling(window).mean()
                    df_roll[f'{col}_roll_std_{window}'] = df_roll[col].rolling(window).std()
                    
        logger.info(f"Created rolling window features")
        
        return df_roll
    
    def create_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create first and second difference features (YoY, QoQ changes).
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with difference features
        """
        logger.info("Creating difference features (changes)...")
        
        df_diff = df.copy()
        numeric_cols = df_diff.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        # First difference (change)
        for col in numeric_cols[:10]:
            df_diff[f'{col}_diff'] = df_diff[col].diff()
            df_diff[f'{col}_pct_change'] = df_diff[col].pct_change()
            
        # Second difference (acceleration)
        for col in numeric_cols[:5]:
            df_diff[f'{col}_diff2'] = df_diff[col].diff().diff()
            
        logger.info("Difference features created")
        
        return df_diff
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend indicators using linear regression.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            DataFrame with trend features
        """
        logger.info("Creating trend features...")
        
        df_trend = df.copy()
        numeric_cols = df_trend.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        for col in numeric_cols[:10]:
            # Simple trend indicator (positive/negative)
            values = df_trend[col].values
            if len(values) > 1:
                x = np.arange(len(values))
                mask = ~np.isnan(values)
                if mask.sum() > 1:
                    slope, _, _, _, _ = stats.linregress(x[mask], values[mask])
                    df_trend[f'{col}_trend_slope'] = slope
                else:
                    df_trend[f'{col}_trend_slope'] = 0
            else:
                df_trend[f'{col}_trend_slope'] = 0
                
        logger.info("Trend features created")
        
        return df_trend
    
    def create_ratio_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create changes in key financial ratios.
        
        Args:
            df: DataFrame with financial ratios
            
        Returns:
            DataFrame with ratio change features
        """
        logger.info("Creating ratio change features...")
        
        df_ratio_change = df.copy()
        
        key_ratios = [
            'current_ratio', 'quick_ratio', 'debt_ratio', 
            'roa', 'roe', 'profit_margin'
        ]
        
        for ratio in key_ratios:
            if ratio in df_ratio_change.columns:
                df_ratio_change[f'{ratio}_yoy_change'] = df_ratio_change[ratio].pct_change()
                df_ratio_change[f'{ratio}_deterioration'] = (
                    df_ratio_change[f'{ratio}_yoy_change'] < 0
                ).astype(int)
                
        logger.info("Ratio change features created")
        
        return df_ratio_change
    
    def create_interaction_features(
        self, 
        df: pd.DataFrame,
        degree: int = None
    ) -> pd.DataFrame:
        """
        Create polynomial and interaction features.
        
        Args:
            df: DataFrame with features
            degree: Polynomial degree
            
        Returns:
            DataFrame with interaction features
        """
        degree = degree or self.config.get('features.polynomial_degree', 2)
        
        if not self.config.get('features.interaction_features', False):
            return df
            
        logger.info(f"Creating interaction features (degree={degree})...")
        
        from sklearn.preprocessing import PolynomialFeatures
        
        df_inter = df.copy()
        numeric_cols = df_inter.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        # Limit to key features to avoid explosion
        key_features = numeric_cols[:5]
        
        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        interactions = poly.fit_transform(df_inter[key_features])
        
        feature_names = poly.get_feature_names_out(key_features)
        interaction_df = pd.DataFrame(
            interactions, 
            columns=feature_names,
            index=df_inter.index
        )
        
        # Add only new interaction columns
        new_cols = [col for col in interaction_df.columns if col not in df_inter.columns]
        df_inter = pd.concat([df_inter, interaction_df[new_cols]], axis=1)
        
        logger.info(f"Created {len(new_cols)} interaction features")
        
        return df_inter
    
    def create_all_econometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all econometric features.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with all econometric features
        """
        logger.info("Creating all econometric features...")
        
        df_econ = df.copy()
        
        # Temporal features
        df_econ = self.create_difference_features(df_econ)
        df_econ = self.create_trend_features(df_econ)
        df_econ = self.create_ratio_changes(df_econ)
        
        # Interaction features
        df_econ = self.create_interaction_features(df_econ)
        
        # CRITICAL: Fill NaN created by differencing operations
        logger.info("Filling NaN values from differencing...")
        numeric_cols = df_econ.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_econ[col].isnull().any():
                # Use forward fill then backward fill
                df_econ[col] = df_econ[col].ffill().bfill().fillna(0)
        
        initial_shape = df_econ.shape
        
        logger.info(f"Econometric features created. Shape: {initial_shape}")
        
        return df_econ