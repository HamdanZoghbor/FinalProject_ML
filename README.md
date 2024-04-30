# FinalProject_ML by Hamdan and Vladimr
# Water Quality Analysis Project

## Introduction
In this project, we develop a framework for assessing water drinking quality using three machine learning models: linear regression, logistic regression, and neural networks. We focus on water pH levels, considering WHO (2007) standards that state most drinking water has a pH ranging from 6.5 to 8.5. This analysis utilizes the *Water Quality Data* dataset provided by Sahir Maharaj (2020).

## Dataset Description

The dataset comprises various water quality measurements:
- Salinity
- Dissolved oxygen levels
- pH levels
- Secchi depth
- Water depth
- Water temperature
- Air temperature

Data points: 2371, with some missing values addressed during data cleaning and preparation.

## Machine Learning Models Used

1. **Linear Regression**: Predicts pH levels using several features, employing regularization techniques like Ridge and Lasso to enhance performance and reduce overfitting. You can also input your own features values and expect a prediction of a given ph score.
2. **Logistic Regression**: Categorizes pH levels into low, medium, or high.
3. **Neural Networks**: Utilized for complex pattern recognition which details are to be specified.

## Installation and Usage

Ensure Python and necessary packages are installed:

```bash
pip install numpy pandas scikit-learn matplotlib
pip install arrow
