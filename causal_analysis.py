"""
Causal Analysis Module for the Educational Inequality Study.
Implements the Double Machine Learning (DML) framework using CausalForestDML
to estimate Conditional Average Treatment Effects (CATEs).
"""

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
import warnings

warnings.filterwarnings('ignore')

class EducationalInequalityAnalyzer:
    """
    Conducts causal analysis of educational inequality using the DML framework.
    """
    
    def __init__(self, random_state: int = 42):
        """Initializes the analyzer."""
        self.random_state = random_state
        self.model = None
        print("EducationalInequalityAnalyzer initialized.")
        
    def _setup_dml_model(self, n_estimators: int = 1000) -> CausalForestDML:
        """
        Initializes the CausalForestDML model with specified parameters.
        These parameters align with modern best practices for DML.
        """
        return CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            model_t=GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            n_estimators=n_estimators,
            random_state=self.random_state,
            criterion='mse',
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5
        )
    
    def _plot_cate_comparison(self, results: Dict, output_path: str) -> None:
        """
        Generates and saves a bar plot comparing CATEs between countries.
        """
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        countries = ['Mexico', 'Chile']
        cates = [results['cate_mexico'], results['cate_chile']]
        errors = [
            [cates[0] - results['ci_mexico'][0], cates[1] - results['ci_chile'][0]],
            [results['ci_mexico'][1] - cates[0], results['ci_chile'][1] - cates[1]]
        ]
        
        bars = ax.bar(countries, cates, yerr=errors,
                      color=['#006847', '#d52b1e'], alpha=0.8,
                      capsize=5, edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        
        ax.set_ylabel('Conditional Average Treatment Effect (CATE)', fontsize=12)
        title = f'CATE of {results["treatment"]} on {results["outcome"]}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        for bar, cate in zip(bars, cates):
            yval = bar.get_height()
            va = 'bottom' if yval >= 0 else 'top'
            offset = 0.05 * max(cates) if yval >= 0 else -0.1 * max(cates)
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{cate:.3f}',
                    ha='center', va=va, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure
        figure_filename = f"cate_{results['scenario'].replace(' ', '_').lower()}.png"
        save_path = os.path.join(output_path, figure_filename)
        plt.savefig(save_path, dpi=300)
        print(f"\nSaved CATE comparison plot to: {save_path}")
        plt.close(fig)

    def run_scenario_analysis(self, 
                            data: pd.DataFrame,
                            scenario_name: str,
                            outcome_var: str,
                            treatment_var: str,
                            heterogeneity_features: List[str],
                            control_vars: List[str],
                            figures_path: str) -> Dict:
        """
        Executes a full DML analysis for a specific scenario.
        """
        print(f"\n{'='*80}")
        print(f"RUNNING SCENARIO: {scenario_name}")
        print(f"{'='*80}")
        print(f"  - Outcome (Y): {outcome_var}")
        print(f"  - Treatment (T): {treatment_var}")
        print(f"  - Heterogeneity Features (X): {heterogeneity_features}")
        print(f"  - Controls (W): {control_vars}")
        
        # Prepare data matrices
        Y = data[outcome_var].values
        T = data[treatment_var].values  
        X = data[heterogeneity_features].values
        W = data[control_vars].values if control_vars else None
        
        # Initialize and fit the DML model
        self.model = self._setup_dml_model()
        self.model.fit(Y, T, X=X, W=W)
        
        # Estimate CATEs and confidence intervals
        cate_estimates = self.model.effect(X)
        cate_intervals = self.model.effect_interval(X, alpha=0.05)
        
        # Calculate country-specific average CATEs
        mexico_mask = data['is_MEX'] == 1
        chile_mask = data['is_MEX'] == 0
        
        results = {
            'scenario': scenario_name,
            'outcome': outcome_var,
            'treatment': treatment_var,
            'cate_mexico': np.mean(cate_estimates[mexico_mask]),
            'cate_chile': np.mean(cate_estimates[chile_mask]),
            'ci_mexico': (np.mean(cate_intervals[0][mexico_mask]), np.mean(cate_intervals[1][mexico_mask])),
            'ci_chile': (np.mean(cate_intervals[0][chile_mask]), np.mean(cate_intervals[1][chile_mask])),
            'n_total': len(data),
            'n_mexico': int(np.sum(mexico_mask)),
            'n_chile': int(np.sum(chile_mask))
        }
        
        # Print results and generate visualization
        self._print_results(results)
        self._plot_cate_comparison(results, figures_path)
        
        return results

    def _print_results(self, results: Dict) -> None:
        """Prints formatted results to the console."""
        print("\n--- Causal Analysis Results ---")
        print(f"Total Observations: {results['n_total']} (Mexico: {results['n_mexico']}, Chile: {results['n_chile']})")
        
        print(f"\nConditional Average Treatment Effects (CATEs):")
        print(f"  - Mexico: {results['cate_mexico']:.4f} (95% CI: [{results['ci_mexico'][0]:.4f}, {results['ci_mexico'][1]:.4f}])")
        print(f"  - Chile:  {results['cate_chile']:.4f} (95% CI: [{results['ci_chile'][0]:.4f}, {results['ci_chile'][1]:.4f}])")
        
        mexico_significant = not (results['ci_mexico'][0] <= 0 <= results['ci_mexico'][1])
        chile_significant = not (results['ci_chile'][0] <= 0 <= results['ci_chile'][1])
        
        print(f"\nStatistical Significance at alpha=0.05:")
        print(f"  - Mexico Effect: {'Significant' if mexico_significant else 'Not Significant'}")
        print(f"  - Chile Effect:  {'Significant' if chile_significant else 'Not Significant'}")

def run_all_scenarios(data: pd.DataFrame, figures_path: str) -> Dict[str, Dict]:
    """
    Executes the three main scenarios from the paper (Table 5).
    """
    analyzer = EducationalInequalityAnalyzer()
    all_results = {}
    
    # Common heterogeneity and control variables from the paper
    HET_FEATURES = ['is_MEX', 'EN.POP.DNST', 'SI.POV.GINI', 'SP.URB.TOTL.IN.ZS']
    CONTROL_VARS = ['NY.GDP.PCAP.PP.KD', 'XGOVEXP.IMF', 'YEARS.FC.COMP.1T3', 'YEARS.FC.FREE.1T3', 'CTRL_SEVERITY']
    
    # Ensure all required columns are present
    required_cols = HET_FEATURES + CONTROL_VARS
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Missing one or more required columns for analysis. Needed: {required_cols}")
        return {}

    # Scenario 1: Population Density -> Max. Intersectional Disadvantage
    all_results['density_disaster'] = analyzer.run_scenario_analysis(
        data=data,
        scenario_name="Density Disaster",
        outcome_var='INT_MAX_DISADVANTAGE',
        treatment_var='EN.POP.DNST',
        heterogeneity_features=[f for f in HET_FEATURES if f != 'EN.POP.DNST'],
        control_vars=[c for c in CONTROL_VARS if c != 'EN.POP.DNST'],
        figures_path=figures_path
    )
    
    # Scenario 2: Government Expenditure -> Rural-Urban Gap
    all_results['convergence_lever'] = analyzer.run_scenario_analysis(
        data=data,
        scenario_name="Convergence Lever",
        outcome_var='RU_OVERAGE_LS_Male',
        treatment_var='XGOVEXP.IMF', 
        heterogeneity_features=[f for f in HET_FEATURES if f != 'XGOVEXP.IMF'],
        control_vars=[c for c in CONTROL_VARS if c != 'XGOVEXP.IMF'],
        figures_path=figures_path
    )
    
    # Scenario 3: Income Inequality -> Max. Intersectional Disadvantage
    all_results['institutional_amplification'] = analyzer.run_scenario_analysis(
        data=data,
        scenario_name="Institutional Amplification",
        outcome_var='INT_MAX_DISADVANTAGE',
        treatment_var='SI.POV.GINI',
        heterogeneity_features=[f for f in HET_FEATURES if f != 'SI.POV.GINI'],
        control_vars=[c for c in CONTROL_VARS if c != 'SI.POV.GINI'],
        figures_path=figures_path
    )
    
    return all_results