"""
Main execution script for the Mexico-Chile Educational Inequality Causal Analysis.

This script orchestrates the full pipeline:
1. Sets up file paths for data and results.
2. Preprocesses the raw panel data using the DataPreprocessor module.
3. Runs the three core causal analysis scenarios from the paper using the
   EducationalInequalityAnalyzer module.
4. Validates the generated results against the key findings reported in the paper.
"""

import os
import pandas as pd
from data_preprocessor import DataPreprocessor
from causal_analysis import run_all_scenarios
from typing import Dict

def validate_paper_results(results: Dict[str, Dict]) -> None:
    """
    Validates that the generated CATEs match the paper's reported values in Table 5.
    """
    print("\n" + "="*80)
    print("VALIDATING RESULTS AGAINST PAPER'S TABLE 5")
    print("="*80)
    
    # Expected results from Table 5 of the paper
    expected_results = {
        'density_disaster': {
            'mexico_cate': 1.02, 'chile_cate': 1.05, 'tolerance': 0.1
        },
        'convergence_lever': {
            'mexico_cate': -0.41, 'chile_cate': -0.34, 'tolerance': 0.1
        },
        'institutional_amplification': {
            'mexico_cate': 1.89, 'chile_cate': 2.94, 'tolerance': 0.2
        }
    }
    
    all_match = True
    for scenario_key, expected in expected_results.items():
        if scenario_key in results:
            actual = results[scenario_key]
            
            print(f"\nValidating Scenario: {actual['scenario']}")
            
            mex_actual = actual['cate_mexico']
            chi_actual = actual['cate_chile']
            mex_expected = expected['mexico_cate']
            chi_expected = expected['chile_cate']
            tolerance = expected['tolerance']
            
            mex_match = abs(mex_actual - mex_expected) <= tolerance
            chi_match = abs(chi_actual - chi_expected) <= tolerance
            
            print(f"  Mexico CATE:  Actual={mex_actual:.2f}, Expected={mex_expected:.2f} -> {'✓ Match' if mex_match else '✗ Mismatch'}")
            print(f"  Chile CATE:   Actual={chi_actual:.2f}, Expected={chi_expected:.2f} -> {'✓ Match' if chi_match else '✗ Mismatch'}")
            
            if not (mex_match and chi_match):
                all_match = False
        else:
            print(f"\nWarning: Scenario '{scenario_key}' not found in results.")
            all_match = False
            
    print("\n" + "-"*80)
    if all_match:
        print("SUCCESS: All generated results match the key findings in the paper.")
    else:
        print("FAILURE: One or more generated results do not match the key findings in the paper.")
    print("-" * 80)

def main():
    """Main function to run the entire analysis pipeline."""
    
    # --- 1. Setup Paths ---
    # Assumes the script is run from the root directory, e.g., `python src/main.py`
    CWD = os.getcwd()
    DATA_PATH = os.path.join(CWD, 'data', 'raw', 'integrated_panel_data.csv')
    PROCESSED_DATA_PATH = os.path.join(CWD, 'data', 'processed', 'processed_data.csv')
    FIGURES_PATH = os.path.join(CWD, 'results', 'figures')
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # --- 2. Preprocess Data ---
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_pipeline(file_path=DATA_PATH)
    
    if processed_data is not None:
        # Save the processed data for inspection and future use
        processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\nProcessed data saved to: {PROCESSED_DATA_PATH}")

        # --- 3. Run Causal Analysis ---
        print("\n" + "="*80)
        print("STARTING CAUSAL ANALYSIS")
        print("="*80)
        analysis_results = run_all_scenarios(processed_data, figures_path=FIGURES_PATH)
        
        # --- 4. Validate Results ---
        if analysis_results:
            validate_paper_results(analysis_results)
    else:
        print("\nHalting execution because data preprocessing failed.")

if __name__ == "__main__":
    main()