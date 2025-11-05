"""Financial ratio calculations and bankruptcy indicators."""

import numpy as np
import pandas as pd
from loguru import logger


class FinancialRatios:
    """Calculate financial ratios and bankruptcy indicators."""
    
    def __init__(self, config):
        """
        Initialize financial ratios calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def calculate_altman_z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Altman Z-Score for bankruptcy prediction.
        
        Original Model:
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        Where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with Z-Score added
        """
        logger.info("Calculating Altman Z-Score...")
        
        df_z = df.copy()
        
        # Try to identify relevant columns (Polish dataset has generic names)
        # Assuming standard financial ratio columns exist
        try:
            # X1: Working Capital / Total Assets
            if 'Attr3' in df.columns:  # Working capital ratio
                X1 = df['Attr3']
            else:
                X1 = 0
                
            # X2: Retained Earnings / Total Assets
            if 'Attr4' in df.columns:
                X2 = df['Attr4']
            else:
                X2 = 0
                
            # X3: EBIT / Total Assets  
            if 'Attr5' in df.columns:
                X3 = df['Attr5']
            else:
                X3 = 0
                
            # X4: Book Value of Equity / Total Liabilities
            if 'Attr6' in df.columns:
                X4 = df['Attr6']
            else:
                X4 = 0
                
            # X5: Sales / Total Assets
            if 'Attr7' in df.columns:
                X5 = df['Attr7']
            else:
                X5 = 0
                
            # Calculate Z-Score
            df_z['altman_z_score'] = (
                1.2 * X1 + 
                1.4 * X2 + 
                3.3 * X3 + 
                0.6 * X4 + 
                1.0 * X5
            )
            
            # Classification based on Z-Score (as numeric indicators)
            df_z['z_score_distress'] = (df_z['altman_z_score'] < 1.8).astype(int)
            df_z['z_score_grey'] = ((df_z['altman_z_score'] >= 1.8) & (df_z['altman_z_score'] < 2.99)).astype(int)
            df_z['z_score_safe'] = (df_z['altman_z_score'] >= 2.99).astype(int)
            
            logger.info("Altman Z-Score calculated successfully")
            
        except Exception as e:
            logger.warning(f"Could not calculate Altman Z-Score: {e}")
            df_z['altman_z_score'] = np.nan
            df_z['z_score_distress'] = 0
            df_z['z_score_grey'] = 0
            df_z['z_score_safe'] = 0
            
        return df_z
    
    def calculate_zmijewski_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Zmijewski Bankruptcy Score.
        
        Model:
        Z = -4.3 - 4.5*X1 + 5.7*X2 - 0.004*X3
        
        Where:
        X1 = Net Income / Total Assets (ROA)
        X2 = Total Debt / Total Assets
        X3 = Current Assets / Current Liabilities
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with Zmijewski score
        """
        logger.info("Calculating Zmijewski Score...")
        
        df_zm = df.copy()
        
        try:
            # X1: ROA
            if 'Attr27' in df.columns:
                X1 = df['Attr27']
            else:
                X1 = 0
                
            # X2: Debt ratio
            if 'Attr31' in df.columns:
                X2 = df['Attr31']
            else:
                X2 = 0
                
            # X3: Current ratio
            if 'Attr1' in df.columns:
                X3 = df['Attr1']
            else:
                X3 = 0
                
            df_zm['zmijewski_score'] = -4.3 - 4.5*X1 + 5.7*X2 - 0.004*X3
            
            logger.info("Zmijewski Score calculated successfully")
            
        except Exception as e:
            logger.warning(f"Could not calculate Zmijewski Score: {e}")
            df_zm['zmijewski_score'] = np.nan
            
        return df_zm
    
    def calculate_liquidity_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity-related ratios.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with liquidity ratios
        """
        logger.info("Calculating liquidity ratios...")
        
        df_liq = df.copy()
        
        # Current ratio (if not already present)
        if 'current_ratio' not in df_liq.columns and 'Attr1' in df.columns:
            df_liq['current_ratio'] = df['Attr1']
            
        # Quick ratio
        if 'quick_ratio' not in df_liq.columns and 'Attr2' in df.columns:
            df_liq['quick_ratio'] = df['Attr2']
            
        # Cash ratio
        if 'cash_ratio' not in df_liq.columns and 'Attr8' in df.columns:
            df_liq['cash_ratio'] = df['Attr8']
            
        # Working capital ratio
        if 'working_capital_ratio' not in df_liq.columns and 'Attr3' in df.columns:
            df_liq['working_capital_ratio'] = df['Attr3']
            
        return df_liq
    
    def calculate_profitability_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate profitability ratios.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with profitability ratios
        """
        logger.info("Calculating profitability ratios...")
        
        df_prof = df.copy()
        
        # ROA
        if 'roa' not in df_prof.columns and 'Attr27' in df.columns:
            df_prof['roa'] = df['Attr27']
            
        # ROE  
        if 'roe' not in df_prof.columns and 'Attr28' in df.columns:
            df_prof['roe'] = df['Attr28']
            
        # Profit margin
        if 'profit_margin' not in df_prof.columns and 'Attr29' in df.columns:
            df_prof['profit_margin'] = df['Attr29']
            
        # Operating margin
        if 'operating_margin' not in df_prof.columns and 'Attr30' in df.columns:
            df_prof['operating_margin'] = df['Attr30']
            
        return df_prof
    
    def calculate_leverage_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate leverage/solvency ratios.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with leverage ratios
        """
        logger.info("Calculating leverage ratios...")
        
        df_lev = df.copy()
        
        # Debt ratio
        if 'debt_ratio' not in df_lev.columns and 'Attr31' in df.columns:
            df_lev['debt_ratio'] = df['Attr31']
            
        # Equity ratio
        if 'equity_ratio' not in df_lev.columns and 'Attr32' in df.columns:
            df_lev['equity_ratio'] = df['Attr32']
            
        # Debt to equity
        if 'debt_to_equity' not in df_lev.columns:
            if 'debt_ratio' in df_lev.columns and 'equity_ratio' in df_lev.columns:
                df_lev['debt_to_equity'] = df_lev['debt_ratio'] / (df_lev['equity_ratio'] + 1e-10)
                
        return df_lev
    
    def calculate_efficiency_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate efficiency/activity ratios.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with efficiency ratios
        """
        logger.info("Calculating efficiency ratios...")
        
        df_eff = df.copy()
        
        # Asset turnover
        if 'asset_turnover' not in df_eff.columns and 'Attr7' in df.columns:
            df_eff['asset_turnover'] = df['Attr7']
            
        # Inventory turnover
        if 'inventory_turnover' not in df_eff.columns and 'Attr9' in df.columns:
            df_eff['inventory_turnover'] = df['Attr9']
            
        # Receivables turnover
        if 'receivables_turnover' not in df_eff.columns and 'Attr10' in df.columns:
            df_eff['receivables_turnover'] = df['Attr10']
            
        return df_eff
    
    def calculate_all_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all financial ratios and indicators.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with all ratios
        """
        logger.info("Calculating all financial ratios...")
        
        df_ratios = df.copy()
        
        # Bankruptcy scores
        if self.config.get('features.altman_z_score', True):
            df_ratios = self.calculate_altman_z_score(df_ratios)
            df_ratios = self.calculate_zmijewski_score(df_ratios)
            
        # Ratio categories
        df_ratios = self.calculate_liquidity_ratios(df_ratios)
        df_ratios = self.calculate_profitability_ratios(df_ratios)
        df_ratios = self.calculate_leverage_ratios(df_ratios)
        df_ratios = self.calculate_efficiency_ratios(df_ratios)
        
        logger.info(f"All financial ratios calculated. Shape: {df_ratios.shape}")
        
        return df_ratios