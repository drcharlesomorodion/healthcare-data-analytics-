# HIV Treatment Adherence Analysis

## Overview

This project analyzes factors affecting antiretroviral therapy (ART) adherence using survival analysis techniques. The analysis is based on data from an HIV treatment program in Nigeria, where I served as HIV Coordinator in collaboration with FHI360.

## Background

Maintaining high adherence to antiretroviral therapy is critical for treatment success and preventing drug resistance. This analysis identifies predictors of treatment discontinuation and develops a risk stratification tool to guide targeted interventions.

## Dataset

The dataset includes:
- **Patient demographics:** Age, Gender
- **Clinical factors:** Baseline CD4 count, Pill burden
- **Program factors:** Adherence support, Distance to clinic
- **Outcome:** Treatment discontinuation (time-to-event)

## Methods

### 1. Descriptive Analysis
- Cohort characteristics
- Follow-up time distribution
- Event rates by subgroups

### 2. Kaplan-Meier Survival Analysis
- Overall treatment retention
- Survival curves by key predictors
- Log-rank tests for group comparisons

### 3. Cox Proportional Hazards Model
- Multivariate analysis of predictors
- Hazard ratios with confidence intervals
- Proportional hazards assumption testing

### 4. Risk Stratification
- Risk score development
- Risk category assignment
- Validation of stratification

## Key Findings

### Predictors of Treatment Discontinuation:
| Factor | Hazard Ratio | 95% CI | p-value |
|--------|-------------|--------|---------|
| Age (per 10 years) | 0.85 | 0.72-1.01 | 0.068 |
| Female (vs Male) | 0.72 | 0.58-0.89 | 0.003 |
| Baseline CD4 <200 | 1.45 | 1.12-1.88 | 0.005 |
| No adherence support | 2.34 | 1.78-3.08 | <0.001 |
| Side effects reported | 1.89 | 1.45-2.46 | <0.001 |
| Distance >15km | 1.67 | 1.28-2.18 | <0.001 |

### Risk Stratification:
- **Low Risk:** 95% retention at 12 months
- **Moderate Risk:** 87% retention at 12 months
- **High Risk:** 72% retention at 12 months

## Files

- `hiv_adherence_analysis.R` - Main analysis script
- `km_survival_by_gender.png` - Survival curves by gender
- `km_survival_by_risk.png` - Survival curves by risk category
- `cox_forest_plot.png` - Hazard ratio forest plot
- `schoenfeld_residuals.png` - Proportional hazards test

## Requirements

```r
# R packages
install.packages("survival")
install.packages("survminer")
install.packages("tidyverse")
install.packages("coxphw")
```

## Usage

```r
source("hiv_adherence_analysis.R")

# Load data
data <- load_hiv_data("hiv_cohort_data.csv")

# Descriptive analysis
descriptive_analysis(data)

# Survival analysis
km_results <- fit_kaplan_meier(data)

# Cox model
cox_model <- fit_cox_model(data)

# Risk stratification
data_with_risk <- create_risk_stratification(data, cox_model)
```

## Clinical Impact

This analysis informed program improvements that achieved:
- 90% treatment adherence rate
- Targeted adherence support for high-risk patients
- Reduced treatment discontinuation by 25%

## Author

**Dr. Charles Osahenrumwen Omorodion**
- HIV Coordinator, Osasco Medical Centre (2015-2018)
- MPH (Distinction) - Arden University, Berlin
- Contact: dr.charlesomo@yahoo.com

## License

MIT License
