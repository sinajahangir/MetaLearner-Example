# MetaLearner-Example
Sample code for developing a meta-learner for hydrological prediction

Streamflow Prediction for Rio Puerco, NM

This project focuses on developing a robust streamflow prediction model for the Rio Puerco basin in New Mexico, leveraging a meta-learning approach that combines multiple prediction methods.

# Methodology

The core of this project is a meta-learning architecture that integrates predictions from a Time Series Foundation Model (TFM) and a Long Short-Term Memory (LSTM) model.

Time Series Foundation Model (TFM): This model uses lagged streamflow observations to produce probabilistic forecasts, including the median, 5%, and 95% quantiles.

LSTM Model: A deep learning model that incorporates both lagged Daymet climate data and static catchment features to generate its streamflow predictions.

Meta-Learner: A three-layer Multi-Layer Perceptron (MLP) acts as the meta-learner. It takes the outputs from both the TFM and the LSTM model to perform bias correction and produce the final, refined streamflow prediction.

This ensemble approach aims to combine the strengths of both models, resulting in more accurate and reliable streamflow forecasts for the region.

# Data

The project utilizes a dataset spanning from 1980 to 2020. The data is split into the following ranges for training, validation, and testing:

Training: 1980-01-01 to 2011-05-13

Validation: 2011-05-13 to 2014-11-06

Testing: 2014-11-06 to 2020-12-31

# Repo

The repo contains sample code for model development and analysis. The original data provided is also shared here. The code for TFM is available: https://github.com/sinajahangir/Foundation-Models

<img width="870" height="871" alt="image" src="https://github.com/user-attachments/assets/f5d8ac26-5b8b-4956-823a-38ac0d610c28" />

