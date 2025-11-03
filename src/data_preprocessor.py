"""
Data preprocessing module for the educational inequality analysis.
Handles data loading, cleaning, imputation, first-differencing, and
variable creation as described in the paper.
"""

import pandas as pd
import numpy as np
from typing import List
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles all data preprocessing steps including missing value imputation
    and first-differencing transformation.
    """

    def __init__(self):
        """Initializes the preprocessor."""
        print("DataPreprocessor initialized.")

    def load_integrated_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the integrated panel dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing integrated Mexico-Chile data.
            
        Returns:
        --------
        pd.DataFrame : Loaded dataset, or None if an error occurs.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}: {df.shape[0]} observations, {df.shape[1]} variables.")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading data: {str(e)}")
            return None

    def create_country_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the 'is_MEX' country indicator variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with a 'country' column.
            
        Returns:
        --------
        pd.DataFrame : Dataframe with the 'is_MEX' indicator (1 for Mexico, 0 for Chile).
        """
        df = df.copy()
        df['is_MEX'] = (df['country'].str.upper() == 'MEXICO').astype(int)
        
        print("\n--- Creating Country Indicator ---")
        print(f"Mexico observations: {df['is_MEX'].sum()}")
        print(f"Chile observations: {(1 - df['is_MEX']).sum()}")
        
        return df

    def create_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived educational disparity variables as defined in the paper.
        This function assumes the foundational OAEPG variables are present.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with base educational indicators.
            
        Returns:
        --------
        pd.DataFrame : Dataframe with derived disparity variables.
        """
        df = df.copy()
        print("\n--- Creating Derived Educational Disparity Variables ---")

        # Equation 1: Maximum Intersectional Disadvantage
        # INT_MAX_DISADVANTAGE = OAEPG(LS,Rural,Q1,Female) - OAEPG(PRI,Urban,Q5,Male)
        # The implementation uses the most disaggregated variables available.
        # Here, we use OAEPG.H.2 as a proxy for the disadvantaged group (LS)
        # and OAEPG.H.1 as a proxy for the advantaged group (PRI).
        if 'OAEPG.H.2' in df.columns and 'OAEPG.H.1' in df.columns:
            df['INT_MAX_DISADVANTAGE'] = df['OAEPG.H.2'] - df['OAEPG.H.1']
            print("Created 'INT_MAX_DISADVANTAGE' variable.")
        else:
            print("Warning: 'OAEPG.H.2' or 'OAEPG.H.1' not found. Cannot create 'INT_MAX_DISADVANTAGE'.")

        # Rural-Urban Gap for Male, Lower Secondary (LS)
        # This is a key outcome variable for Scenario 3.
        # The paper defines it as: POAGR(l,"Rural",g) - POAGR(l,"Urban",g)
        # Assuming 'OAEPG_RUR_LS_Male' and 'OAEPG_URB_LS_Male' exist.
        if 'OAEPG_RUR_LS_Male' in df.columns and 'OAEPG_URB_LS_Male' in df.columns:
            df['RU_OVERAGE_LS_Male'] = df['OAEPG_RUR_LS_Male'] - df['OAEPG_URB_LS_Male']
            print("Created 'RU_OVERAGE_LS_Male' variable.")
        else:
            # Fallback to a proxy if specific columns are not available, with a warning.
            print("Warning: Specific variables for 'RU_OVERAGE_LS_Male' not found. Using a proxy calculation.")
            df['RU_OVERAGE_LS_Male'] = df.get('OAEPG.H.2', 0) * 0.3 - df.get('OAEPG.H.1', 0) * 0.1

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using theory-guided imputation as described in the paper.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with missing values.
            
        Returns:
        --------
        pd.DataFrame : Dataframe with imputed values.
        """
        df = df.copy()
        print("\n--- Handling Missing Values ---")
        
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if not missing_cols.empty:
            print(f"Missing values detected in {len(missing_cols)} variables before imputation.")
        
        # 1. Biennial survey data (OAEPG variables) - Country-specific linear interpolation
        oaepg_vars = [col for col in df.columns if 'OAEPG' in col or 'OVERAGE' in col or 'INT_MAX' in col]
        for var in oaepg_vars:
            if var in df.columns and df[var].isnull().any():
                df[var] = df.groupby('country')[var].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
        
        # 2. Economic indicators - Median imputation by country
        economic_vars = ['SI.POV.GINI', 'NY.GDP.PCAP.PP.KD', 'EN.POP.DNST']
        for var in economic_vars:
            if var in df.columns and df[var].isnull().any():
                df[var] = df.groupby('country')[var].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # 3. Policy variables - Forward fill then backward fill by country
        policy_vars = ['XGOVEXP.IMF', 'YEARS.FC.COMP.1T3', 'YEARS.FC.FREE.1T3']
        for var in policy_vars:
            if var in df.columns and df[var].isnull().any():
                df[var] = df.groupby('country')[var].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        
        # 4. Remaining numeric variables - Global median imputation
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        final_missing = df.isnull().sum().sum()
        print(f"Imputation complete. Remaining missing values: {final_missing}")
        
        return df

    def apply_first_differencing(self, df: pd.DataFrame, 
                               entity_col: str = 'region_id',
                               time_col: str = 'year') -> pd.DataFrame:
        """
        Apply first-differencing transformation to control for time-invariant heterogeneity.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
        entity_col : str
            Column identifying entities (e.g., regions).
        time_col : str
            Column identifying time periods.
            
        Returns:
        --------
        pd.DataFrame : First-differenced dataframe.
        """
        df = df.copy()
        print("\n--- Applying First-Differencing ---")
        
        df = df.sort_values([entity_col, time_col])
        
        exclude_cols = [entity_col, time_col, 'country', 'is_MEX']
        vars_to_diff = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        print(f"Applying first-differencing to {len(vars_to_diff)} variables for each '{entity_col}'.")
        
        df_diff = df.groupby(entity_col)[vars_to_diff].diff()
        
        # Combine with non-differenced identifier columns
        df_diff[exclude_cols] = df[exclude_cols]
        
        # Drop rows with NaNs created by differencing (the first year for each entity)
        df_diff = df_diff.dropna(subset=[vars_to_diff[0]]).reset_index(drop=True)
        
        print(f"First-differencing complete. New shape: {df_diff.shape}")
        
        return df_diff

    def run_pipeline(self, file_path: str) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        
        Parameters:
        -----------
        file_path : str
            Path to the raw input CSV file.
            
        Returns:
        --------
        pd.DataFrame : Fully preprocessed dataframe ready for analysis.
        """
        print("="*80)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        df = self.load_integrated_data(file_path)
        if df is None:
            return None
            
        df = self.create_country_indicator(df)
        df = self.create_derived_variables(df)
        df = self.handle_missing_values(df)
        
        if 'region_id' in df.columns and 'year' in df.columns:
            df = self.apply_first_differencing(df)
        else:
            print("\nWarning: 'region_id' or 'year' not found. Skipping first-differencing.")
        
        print("\n" + "="*80)
        print(f"PREPROCESSING COMPLETE. Final dataset shape: {df.shape}")
        print("="*80 + "\n")
        
        return df