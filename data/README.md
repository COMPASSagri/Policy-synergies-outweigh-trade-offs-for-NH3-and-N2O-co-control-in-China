# Data Description

This folder contains the datasets used to train and evaluate the NH3 and N2O emission factor models, as well as datasets used to estimate the NH3 and N2O emissions.

## Contents

- nh3XY.xlsx: Dataset used for training and evaluating the NH3 emission factor model  
- n2oXY.xlsx: Dataset used for training and evaluating the N2O emission factor model
- RF_grid_nh3.pkl: Trained random forest model for NH3 emission factor estimation  
- RF_grid_n2o.pkl: Trained random forest model for N2O emission factor estimation
- CV_factors.xlsx: Coefficients of variation (CV) assigned to different emission sources, used for uncertainty analysis.

## Notes

- The datasets represent compiled and preprocessed data used for machine learning model development.  
- Data splitting (training/testing) is implemented in the code.  

## Reference

For detailed data sources and processing methods, please refer to the associated paper.
