# Cardiovascular Disease Risk Prediction

## Overview

This project implements machine learning models to predict cardiovascular disease (CVD) risk based on patient demographics, clinical indicators, and lifestyle factors. The goal is to develop an interpretable risk prediction tool that can support clinical decision-making.

## Dataset

The dataset includes the following features:
- **Demographics:** Age, Gender, Height, Weight
- **Clinical Measures:** Systolic BP (ap_hi), Diastolic BP (ap_lo), Cholesterol, Glucose
- **Lifestyle Factors:** Smoking, Alcohol intake, Physical activity
- **Target:** Presence of cardiovascular disease (binary)

## Methods

### 1. Exploratory Data Analysis
- Distribution analysis of key risk factors
- Correlation analysis
- Data quality assessment

### 2. Feature Engineering
- BMI calculation
- Age conversion to years
- Blood pressure categorization

### 3. Machine Learning Models
- **Logistic Regression:** Interpretable baseline model
- **Random Forest:** Ensemble method with feature importance

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-score
- AUC-ROC analysis
- Cross-validation
- Feature importance analysis

## Key Results

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 0.73 | 0.79 |
| Random Forest | 0.87 | 0.92 |

### Top Risk Factors (by importance):
1. Age
2. Systolic Blood Pressure
3. BMI
4. Cholesterol Level
5. Weight

## Files

- `cardiovascular_risk_prediction.py` - Main analysis script
- `cardiovascular_eda.png` - Exploratory data analysis visualizations
- `feature_importance.png` - Feature importance plot
- `roc_comparison.png` - ROC curve comparison

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage

```python
from cardiovascular_risk_prediction import CardiovascularRiskPredictor

# Initialize predictor
predictor = CardiovascularRiskPredictor()

# Load data
predictor.load_data('cardio_data.csv')

# Explore and visualize
predictor.explore_data()
predictor.visualize_data()

# Preprocess and train
predictor.preprocess_data()
predictor.train_logistic_regression()
predictor.train_random_forest()

# Compare models
predictor.compare_models()

# Predict for new patients
risk = predictor.predict_risk(new_patient_data)
```

## Clinical Applications

This model can be used for:
- Early identification of high-risk patients
- Guiding preventive interventions
- Patient education and counseling
- Population health screening programs

## Author

**Dr. Charles Osahenrumwen Omorodion**
- Medical Doctor & Public Health Professional
- MPH (Distinction) - Arden University, Berlin
- Contact: dr.charlesomo@yahoo.com

## License

MIT License
