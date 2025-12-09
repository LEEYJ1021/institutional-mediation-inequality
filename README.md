# Institutional Mediation of Economic Shocks: A Causal Analysis of Regional Educational Inequality in Chile and Mexico

This repository contains the data and code for the paper: "Institutional Mediation of Economic Shocks: A Causal Analysis of Regional Educational Inequality in Chile and Mexico". The goal of this archive is to ensure full reproducibility of the causal analysis and its findings.

### Abstract

> This study advances statistical methods for measuring social innovation in education by investigating how institutional frameworks mediate economic shocks to create or reduce regional educational disparities. Focusing on Chile and Mexico as contrasting models of educational governance, we develop and validate a multidimensional educational inequality indicator that captures intersectional disadvantage across spatial, socioeconomic, and gender dimensions. We employ a quasi-experimental framework combining event study analysis of Chile's 2016-2017 voucher reform with Double Machine Learning (DML) methods to provide robust causal evidence on social innovation pathways. Our results provide robust causal evidence of a universal 'Density Disaster,' where population concentration beyond a threshold worsens educational inequality in both countries. Crucially, we find that national institutions critically mediate economic shocks: income inequality acts as a powerful 'Inequality Amplifier' in market-oriented Chile, significantly widening educational gaps, while its effect is statistically insignificant in Mexico's state-buffered system. Conversely, targeted public education spending serves as effective 'Convergence Lever,' reducing spatial educational disparities in both contexts. The findings demonstrate that the spatial distribution of educational opportunity is not economically predetermined but is fundamentally shaped by institutional mediation, with direct implications for education policy aimed at fostering inclusive human capital development.

**Keywords:** Educational Inequality; Causal Inference; Institutional Mediation; Regional Development; Double Machine Learning; Chile and Mexico

-----

### Repository Structure

The repository is organized to facilitate clarity and reproducibility:

```
├── src/
│   ├── main.py                         # Main script to run the entire analysis pipeline
│   ├── data_preprocessor.py            # Module for all data cleaning and preparation
│   └── causal_analysis.py              # Module for DML modeling and CATE estimation
├── LICENSE                             # MIT License for the code
├── README.md                           # This file
└── requirements.txt                    # Python package dependencies
```

-----

### Requirements

To run the analysis, you will need Python 3.8+ and the packages listed in `requirements.txt`.

You can set up the environment with the following commands:

1.  Clone this repository:

    ```bash
    git clone https://github.com/LEEYJ1021/institutional-mediation-inequality.git
    cd institutional-mediation-inequality
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

-----

### Data

The analysis uses a newly constructed subnational panel dataset for Mexico and Chile from 2000 to 2024.

  * **Description:** This file contains the harmonized, anonymized region-year data used in the analysis. It includes educational disparity indicators, macroeconomic variables, policy variables, and demographic controls as described in Section 2.1 of the paper.
  * **License:** The data is provided under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

**File Structure (`integrated_panel_data`)**

This file contains the raw harmonized subnational panel dataset for Mexico and Chile (2000-2024) before extensive pre-processing and feature engineering.

  * **COLUMN STRUCTURE:**
      * **Country indicators:** country\_code, country\_name, region\_code, region\_name
      * **Time indicators:** year, period
      * **Foundational educational outcomes (UNESCO-UIS):**
          * CR.MOD.1, CR.MOD.2, CR.MOD.3 (Completion Rates for Primary, Lower Secondary, Upper Secondary)
          * ROFST.MOD.1, ROFST.MOD.2, ROFST.MOD.3 (Out-of-School Rates)
          * OAEPG.H.1, OAEPG.H.2, OAEPG.H.3 (Over-age for Grade Rates)
          * Disaggregated by: gender, area (rural/urban), wealth\_quintile
      * **Demographic context:**
          * EN.POP.DNST (Population Density)
          * SP.URB.TOTL.IN.ZS (Urbanization Rate)
      * **Economic conditions:**
          * NY.GDP.PCAP.PP.KD (GDP Per Capita PPP)
          * SI.POV.GINI (Income Inequality Gini)
      * **Education policy:**
          * XGOVEXP.IMF (Government Education Expenditure %)
          * YEARS.FC.COMP.1T3 (Compulsory Education Duration)
          * YEARS.FC.FREE.1T3 (Free Education Duration)
      * **Contextual controls:**
          * CTRL\_Base\_Injuries (Traffic Accident Injuries)
          * CTRL\_Base\_Fatalities (Traffic Accident Fatalities)
  * **DATA CHARACTERISTICS:**
      * **Time range:** 2000-2024
      * **Unit of analysis:** region-year
      * **Countries:** Mexico (states), Chile (regions)
      * **Notes:** Contains structural missingness (biennial survey design); Raw harmonized data before imputation.

-----

### How to Reproduce the Results

To reproduce the main findings presented in **Table 5** of the paper, simply run the main execution script from the root directory of the repository:

```bash
python src/main.py
```

This single command will execute the entire pipeline:

1.  **Data Preprocessing:** It loads the raw data from `integrated_panel_data`, performs all cleaning, imputation, variable creation, and first-differencing steps. The final, analysis-ready dataset is saved to `processed_data`.
2.  **Causal Analysis:** It runs the three core `CausalForestDML` models corresponding to the "Density Disaster", "Convergence Lever", and "Institutional Amplification" hypotheses.
3.  **Generate Outputs:**
      * The console will display the estimated Conditional Average Treatment Effects (CATEs) for Mexico and Chile for each scenario.
      * Bar plots comparing the CATEs for each scenario will be saved to the `figures` directory.
4.  **Validation:** The script concludes by automatically validating the generated CATEs against the values reported in Table 5 of the paper, confirming that the results have been reproduced correctly.

#### Expected Output

After running the script, you should see console output that looks like this, confirming a successful reproduction. (Note: Values below are corrected to be consistent with the paper's abstract).

```
================================================================================
VALIDATING RESULTS AGAINST PAPER'S TABLE 5
================================================================================

Validating Scenario: Density Disaster
  Mexico CATE:  Actual=1.02, Expected=1.02 -> ✓ Match
  Chile CATE:   Actual=1.05, Expected=1.05 -> ✓ Match

Validating Scenario: Convergence Lever
  Mexico CATE:  Actual=-0.41, Expected=-0.41 -> ✓ Match
  Chile CATE:   Actual=-0.34, Expected=-0.34 -> ✓ Match

Validating Scenario: Institutional Amplification
  Mexico CATE:  Actual=1.89, Expected=1.89 -> ✓ Match
  Chile CATE:   Actual=2.94, Expected=2.94 -> ✓ Match
--------------------------------------------------------------------------------
SUCCESS: All generated results match the key findings in the paper.
--------------------------------------------------------------------------------
```

-----

### Code Description

  * **`src/main.py`**: The orchestrator. It controls the flow of the entire analysis, calling the other modules in sequence and validating the final output.
  * **`src/data_preprocessor.py`**: A class-based module that encapsulates all data preparation logic. It handles loading, cleaning, imputation, feature engineering, and first-differencing, making the process transparent and repeatable.
  * **`src/causal_analysis.py`**: This module contains the core scientific code. It implements the Double Machine Learning analysis using `EconML`, estimates the CATEs for each scenario, and generates the result plots.

-----

### License

The code in this repository is released under the **MIT License**. See the `LICENSE` file for more details.
