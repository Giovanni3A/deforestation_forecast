# Deforestation Prediction with Deep Learning

## Overview
This repository contains code and Jupyter notebooks for data processing, deep learning model training, and evaluation in the context of deforestation prediction using satellite images in the Brazilian Amazon.

## Data Processing
Data processing notebooks are located in the `notebooks/preprocess` directory. Each notebook loads data from multiple data sources to create a dataset used as model input.

## Exploratory Data Analysis
Some exploratory data analysis notebooks are located in the `notebooks/eda` directory. The notebooks investigate the relationship between variables of interest.

## Model Training
In the `notebooks/models` directory you can find the model code and model training / evaluation notebook.

## Configuration
The `config.py` file declares relevant variables for the whole pipeline, such as data folder path, timestamp limits and data processing parameters.

## Usage
To use the conde and notebooks in this repository, follow these steps:

1. Clone the repository: `git clone https://github.com/Giovanni3A/DeforestationForecast.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download necessary data.
4. Declare the local data folder path as `DATA_PATH` variable in config file.
5. Run the `deter` data processing notebook.
6. Run other data processing notebooks.
7. Run the `resunet_class` notebook to train and evaluate the model.