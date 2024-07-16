# Laptop Price Prediction using Machine Learning

This project involves predicting the price of laptops based on various attributes using machine learning models. The dataset includes features such as CPU, RAM, screen size, and more.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Requirements](#requirements)

## Project Description
The objective of this project is to build a machine learning model to predict the `buynow_price` of laptops. The models are evaluated based on the Root Mean Squared Error (RMSE).

## Dataset
The dataset is divided into training, validation, and test sets, stored in JSON format:
- `train_dataset.json`
- `val_dataset.json`
- `test_dataset.json`

## Data Preprocessing
- **Cleaning:** Removed missing values and irrelevant columns.
- **Feature Engineering:** Split multi-valued columns, encoded categorical variables, and handled multicollinearity using Variance Inflation Factor (VIF).

## Modeling
Implemented and compared various models:
- **Classical Machine Learning:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, and MLP.
- **Ensemble Techniques:** Voting Regressor combining Gradient Boosting, Extra Trees, and Random Forest.
- **Deep Learning:** Neural network models using TensorFlow and Keras.

## Results
The best model was a Voting Regressor with an RMSE of 504.99.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- tensorflow
- keras
