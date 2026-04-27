# Code Description

This folder contains scripts and trained models for estimating NH3 and N2O emission factors, as well as calculating emissions and their uncertainty ranges.

## Contents

### Model Training

- nh3_ml_training.py: Script for training the NH3 emission factor model using machine learning  
- n2o_ml_training.py: Script for training the N2O emission factor model using machine learning  

### Emission Estimation

- fertilizer_nh3_emissions.py: Predict NH3 emission factors and estimate total emissions and their confidence intervals  
- fertilizer_n2o_emissions.py: Predict N2O emission factors and estimate total emissions and their confidence intervals  

### Uncertainty Analysis

- generate_CV.py: Generate multiplicative factors based on coefficients of variation (CV) for uncertainty propagation  

### Trained Models

- RF_grid_nh3.pkl: Trained random forest model for NH3 emission factor estimation  
- RF_grid_n2o.pkl: Trained random forest model for N2O emission factor estimation  

## Workflow

1. Train models using:
   - nh3_ml_training.py  
   - n2o_ml_training.py  

2. Generate uncertainty factors:
   - generate_CV.py  

3. Estimate emissions and uncertainty:
   - fertilizer_nh3_emissions.py  
   - fertilizer_n2o_emissions.py  

## Requirements

- Python 3.x  
- numpy  
- pandas  
- scikit-learn  
- joblib  

## Notes

- Input data should follow the format provided in the `data/` folder.  
- The trained models can be directly loaded for prediction without retraining.  
- Uncertainty ranges are estimated using CV-based multiplicative factors.

## Reference

For detailed methodology, model evaluation, and uncertainty analysis, please refer to the associated paper.
