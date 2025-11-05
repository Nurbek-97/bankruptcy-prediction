"""Main feature engineering pipeline."""

import numpy as np
import pandas as pd
from loguru import logger

from src.features.financial_ratios import FinancialRatios
from src.features.econometric_features import EconometricFeatures


class FeatureEngineer:
    """Main feature engineering orchestrator."""
    
    def __init__(self, config):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.financial_ratios = FinancialRatios(config)
        self.econometric_features = EconometricFeatures(config)
        
    def engineer_features(
        self, 
        df: pd.DataFrame,
        create_ratios: bool = True,
        create_econometric: bool = True
    ) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            create_ratios: Whether to create financial ratios
            create_econometric: Whether to create econometric features
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        df_eng = df.copy()
        initial_features = len(df_eng.columns)
        
        # Financial ratios
        if create_ratios and self.config.get('features.create_ratios', True):
            df_eng = self.financial_ratios.calculate_all_ratios(df_eng)
            
        # Econometric features
        if create_econometric:
            df_eng = self.econometric_features.create_all_econometric_features(df_eng)
            
        # Handle infinite values
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        
        final_features = len(df_eng.columns)
        logger.info(f"Feature engineering completed: {initial_features} -> {final_features} features")
        
        return df_eng
    
    def create_industry_adjusted_features(
        self,
        df: pd.DataFrame,
        industry_col: str = 'industry'
    ) -> pd.DataFrame:
        """
        Create industry-adjusted features (z-scores within industry).
        
        Args:
            df: DataFrame with industry column
            industry_col: Name of industry column
            
        Returns:
            DataFrame with industry-adjusted features
        """
        if industry_col not in df.columns:
            logger.warning(f"Industry column '{industry_col}' not found, skipping adjustment")
            return df
            
        logger.info("Creating industry-adjusted features...")
        
        df_adj = df.copy()
        numeric_cols = df_adj.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        for col in numeric_cols[:10]:
            # Calculate industry mean and std
            industry_stats = df_adj.groupby(industry_col)[col].agg(['mean', 'std'])
            
            # Create adjusted feature (z-score within industry)
            df_adj[f'{col}_industry_adj'] = df_adj.apply(
                lambda row: (
                    (row[col] - industry_stats.loc[row[industry_col], 'mean']) / 
                    (industry_stats.loc[row[industry_col], 'std'] + 1e-10)
                ) if row[industry_col] in industry_stats.index else 0,
                axis=1
            )
            
        logger.info("Industry-adjusted features created")
        
        return df_adj
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated features from existing features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with aggregated features
        """
        logger.info("Creating aggregated features...")
        
        df_agg = df.copy()
        
        # Liquidity score (average of liquidity ratios)
        liquidity_cols = [col for col in df_agg.columns if 'ratio' in col.lower() or 'liquid' in col.lower()]
        if liquidity_cols:
            df_agg['liquidity_score'] = df_agg[liquidity_cols[:5]].mean(axis=1)
            
        # Profitability score
        profitability_cols = [col for col in df_agg.columns if 'roa' in col.lower() or 'roe' in col.lower() or 'profit' in col.lower()]
        if profitability_cols:
            df_agg['profitability_score'] = df_agg[profitability_cols[:5]].mean(axis=1)
            
        # Leverage score
        leverage_cols = [col for col in df_agg.columns if 'debt' in col.lower() or 'leverage' in col.lower()]
        if leverage_cols:
            df_agg['leverage_score'] = df_agg[leverage_cols[:5]].mean(axis=1)
            
        # Overall financial health score
        score_cols = ['liquidity_score', 'profitability_score']
        available_scores = [col for col in score_cols if col in df_agg.columns]
        
        if available_scores:
            df_agg['financial_health_score'] = df_agg[available_scores].mean(axis=1)
            if 'leverage_score' in df_agg.columns:
                df_agg['financial_health_score'] -= df_agg['leverage_score'] * 0.3
                
        logger.info("Aggregated features created")
        
        return df_agg
    
    def get_feature_importance_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simple feature importance proxy based on correlation with target.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            DataFrame with feature importance scores
        """
        if 'class' not in df.columns:
            logger.warning("Target 'class' not found, cannot calculate importance")
            return pd.DataFrame()
            
        logger.info("Calculating feature importance proxy...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['class', 'year']]
        
        correlations = df[numeric_cols].corrwith(df['class']).abs()
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 features by correlation:\n{importance_df.head(10)}")
        
        return importance_df