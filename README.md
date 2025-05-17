# Detecting Single Point Attacks in Electricity Markets Using Decision Trees

This repository contains code and resources for detecting market manipulation attacks in electricity markets using machine learning. The project simulates various types of attacks on electricity markets, engineers features to detect these attacks, and evaluates multiple machine learning models.

## Project Overview

Electricity market operators face risks from errors and deliberate attacks that can manipulate dispatch decisions and inflate costs. This project develops a machine learning framework to detect single-point market anomalies using synthetic data generated from power system simulations.

We simulate four types of attacks targeting generator parameters:
1. Ramp Rate manipulation
2. Upper generation limit alterations
3. Lower generation limit alterations
4. Cost coefficient modifications

## Repository Structure

- `/Cases`: Power system test cases used for simulation
- `/OPF`: Optimal Power Flow code and simulation framework 
- `/dfigures`: Data visualization figures
- `/models`: Saved machine learning models
- `/session_figures`: Seasonal model analysis figures
- `01_data_EDA.ipynb`: Data exploration and preprocessing
- `02_Model_Training.ipynb`: Training and evaluation on full dataset
- `03_Seasonal_Model_Training.ipynb`: Training and evaluation on seasonal subsets
- `advanced_features.py`: Feature engineering functions

## Data

Due to size limitations, the data files are not included in this repository. You can access the full dataset here:
[https://drive.google.com/drive/folders/111mBdrUh0sm6b1xIoCIDCTfasVNjbBax?usp=sharing](https://drive.google.com/drive/folders/111mBdrUh0sm6b1xIoCIDCTfasVNjbBax?usp=sharing)

## Getting Started

### Prerequisites

- MATLAB (for OPF simulations)
- Python 3.8+
- Required Python packages: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

### Workflow

1. **Data Generation**: Run the MATLAB files in the `/OPF` directory to generate synthetic power system data
   - The main function is `spaOPFsmlF.m` which simulates normal operation and attacks

2. **Data Preprocessing**: Run `01_data_EDA.ipynb` to:
   - Load and clean the raw data
   - Perform exploratory data analysis
   - Prepare the data for model training

3. **Full Dataset Model Training**: Run `02_Model_Training.ipynb` to:
   - Engineer features (including automated feature generation from `advanced_features.py`)
   - Train multiple models (Extra Trees, Random Forest, Gradient Boosting, XGBoost, etc.)
   - Evaluate model performance
   - Create a Voting Classifier ensemble of the best models

4. **Seasonal Model Training**: Run `03_Seasonal_Model_Training.ipynb` to:
   - Train separate models for each season
   - Compare performance across different seasonal contexts
   - Analyze feature importance variations

## Key Results

- Best performing model on both full dataset and seasonal subsets: Voting Classifier, Gradient Boosting, XGBoost (100% accuracy) UPDATED!


## License

This project is for educational and research purposes.

## Acknowledgments

- This research was inspired by real market anomalies like the Ontario IESO phantom demand incident
- Power Grid Library (PGLib) for providing the IEEE 300-bus test case
