# **Google Stock Price Prediction using Prophet and Bayesian Optimization**

This repository contains a Jupyter Notebook that demonstrates how to predict Google stock prices using the Prophet time series forecasting library and optimize its hyperparameters using Bayesian Optimization.

## **Table of Contents**

* [Project Description](#project-description)  
* [Dataset](#dataset)  
* [Methodology](#methodology)  
* [Hyperparameter Tuning](#hyperparameter-tuning)  
* [Getting Started](#getting-started)  
  * [Prerequisites](#prerequisites)  
  * [Installation](#installation)  
  * [Running the Code](#running-the-code)  
* [Results](#results)  
* [Visualizations](#visualizations)  


## **Project Description**

This project aims to forecast future closing prices of Google stock using historical data. It leverages Facebook's Prophet library, which is designed for analyzing time series data with strong seasonal effects and historical data that is not necessarily regular. To improve the model's performance, Bayesian Optimization is employed to find the optimal combination of key Prophet hyperparameters.

## **Dataset**

The project uses historical Google stock price data, provided in a CSV file named GOOGLE.csv. The dataset is expected to contain at least the following columns:

* Date: The date of the stock price record.  
* Close: The closing price of the stock on that date.

The notebook performs basic data loading and checks for missing or duplicate values.

## **Methodology**

The core of the prediction model is the **Prophet** library. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

The steps involved are:

1. Load the historical stock data.  
2. Prepare the data by renaming the 'Date' column to 'ds' and the 'Close' column to 'y', as required by Prophet.  
3. Split the data into training and testing sets (80% for training, 20% for testing).  
4. Define a function that trains a Prophet model with specified hyperparameters and evaluates its performance (using Mean Absolute Error \- MAE) on the test set.  
5. Use **Bayesian Optimization** to find the set of hyperparameters for the Prophet model that minimizes the MAE on the test set.

## **Hyperparameter Tuning**

Bayesian Optimization is used to efficiently search the hyperparameter space for the Prophet model. The hyperparameters tuned are:

* changepoint\_prior\_scale: This parameter determines the flexibility of the trend, specifically how much the trend is allowed to change at the changepoints.  
* seasonality\_prior\_scale: This parameter adjusts the strength of the seasonality model.  
* holidays\_prior\_scale: This parameter adjusts the strength of the holiday components.

The optimization process explores different combinations of these parameters within defined bounds to find the set that yields the lowest Mean Absolute Error (MAE) on the test data.

## **Getting Started**

These instructions will get you a copy of the project up and running on your local machine or a cloud environment like Google Colab.

### **Prerequisites**

* Python 3.x  
* pandas  
* numpy  
* matplotlib  
* prophet  
* scikit-learn  
* bayesian-optimization

### **Installation**

You can install the necessary libraries using pip:

pip install pandas numpy matplotlib prophet scikit-learn bayesian-optimization

If you are using Google Colab, you may only need to install bayesian-optimization and prophet as others are usually pre-installed:

\!pip install bayesian-optimization prophet

### **Running the Code**

1. Download the GOOGLE.csv dataset and the Jupyter Notebook (“Google\_Stock\_Price\_Prediction”.ipynb).  
2. Upload both files to your Google Colab environment or ensure they are in the same directory if running locally.  
3. Open the notebook in Google Colab or your local Jupyter environment.  
4. Run the cells sequentially. The notebook will:  
   * Load and preprocess the data.  
   * Define the model evaluation function for Bayesian Optimization.  
   * Run the Bayesian Optimization process to find the best hyperparameters.  
   * Train the final Prophet model using the best hyperparameters on the training data.  
   * Make future predictions.  
   * Plot the results.

## **Results**

The notebook outputs the results of the Bayesian Optimization, showing the exploration of different hyperparameters and the corresponding target values (negative MAE). It identifies and prints the best hyperparameters found.

Example output of best hyperparameters:

Best Hyperparameters: {'changepoint\_prior\_scale': 0.029983722471931533, 'holiday\_prior\_scale': 8.663099696291603, 'seasonality\_prior\_scale': 6.015138967314656}

It also prints the tail of the forecast dataframe, showing the predicted stock prices (yhat) along with the uncertainty intervals (yhat\_lower, yhat\_upper).

Example forecast output:

             ds         yhat   yhat\_lower   yhat\_upper  
4426 2021-02-13  1605.322841  1401.741009  1826.535107  
4427 2021-02-14  1605.888683  1410.561836  1830.971235  
4428 2021-02-15  1610.009263  1411.350682  1835.679070  
4429 2021-02-16  1610.579527  1408.867511  1826.485664  
4430 2021-02-17  1611.596775  1406.061178  1833.536811

## **Visualizations**

The notebook generates a plot showing the historical training data, the actual test data, and the Prophet model's forecast. This visualization helps assess how well the model captures the trends and seasonality in the stock prices and how closely the predictions align with the actual values.
