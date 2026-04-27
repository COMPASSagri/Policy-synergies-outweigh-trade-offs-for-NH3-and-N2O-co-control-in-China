# Policy-synergies-outweigh-trade-offs-for-NH3-and-N2O-co-control-in-China
Code repository for the paper : "Policy synergies outweigh trade-offs for NH3 and N2O co-control in China"
# Model Card – NH3 and N2O Emission Factors from Cropland

## Model Details

This repository provides a machine learning-based model for estimating cropland NH3 and N2O emission factors during the fertilization process, along with an uncertainty quantification framework to estimate the confidence intervals of the resulting emissions.

The model is designed to capture the nonlinear relationships between emission factors and multiple environmental and management variables, including:

- Climate conditions  
- Soil properties  
- Crop types  
- Irrigation practices  
- Fertilization and tillage management  

This work extends previous efforts on NH3 emission estimation by incorporating N2O emission factors within a unified modeling framework.

## Intended Use

This model is intended for:

- Supporting sustainable agricultural management (e.g., fertilizer optimization, emission mitigation)  
- Quantifying agricultural reactive nitrogen emissions  
- Assisting policy-making related to air quality and climate change  
- Advancing research in earth system and environmental sciences  


## Input Factors

The model considers multiple driving factors, including:

- Meteorological variables  
- Soil characteristics  
- Crop categories  
- Irrigation water inputs  
- Fertilizer application rates and methods  
- Tillage practices  

## Metrics

Model performance is evaluated using:

- Coefficient of determination (R²)  
- Root Mean Square Error (RMSE)  

These metrics quantify the agreement between observed and predicted emission factors.

## Training Data

- 80% of the compiled dataset is used for model training  

## Evaluation Data

- 20% of the dataset is reserved for independent evaluation  

## Data Availability

The training dataset used in this study is partially provided in this repository. Additional data may be made available upon reasonable request, subject to data sharing policies.

## Ethical Considerations

The model highlights the potential of data-driven approaches to improve nitrogen management and reduce NH3 and N2O emissions without necessarily increasing fertilizer inputs. 

However, results should be interpreted carefully to avoid overgeneralization beyond the supported conditions.

## Caveats and Recommendations

Due to data limitations, several potentially important factors are not explicitly included, such as:

- Enhanced-efficiency fertilizers (e.g., inhibitors, coated fertilizers)  
- Organic manure characteristics  
- Irrigation techniques and scheduling  
- Timing of precipitation events  
- Fertilizer incorporation depth  

Future work could incorporate these factors to further improve model robustness and generalizability.
