# Credit Card Fraud Detection

## Overview
This project implements a Credit Card Fraud Detection system using machine learning techniques. The system identifies fraudulent transactions by analyzing transaction data and applying a Random Forest classifier. It uses techniques like class balancing to improve the model's performance.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)

## Features
- **Feature Selection:** Utilizes Random Forest to determine and select important features.
- **Class Balancing:** Implements SMOTE to balance the dataset for improved model performance.
- **Model Training:** Trains a Random Forest classifier and uses Grid Search for hyperparameter tuning.
- **API:** Provides a Flask-based REST API for making predictions.
- **High Accuracy:** The model achieves an accuracy score of **0.99**, indicating excellent performance in detecting fraudulent transactions.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (for SMOTE)
- Flask
  
## Dataset
The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders, where the positive class (fraudulent transactions) is a minority class.

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Kerissa23/Credit-Card-Fraud-Detection.git
   cd credit-card-fraud-detection

## Usage
1. Data Preprocessing:
   Run the preprocess.py to preprocess the dataset and save the processed data:
   ```bash
   python preprocess.py
2. Model Training:
   Run the model.py to train the model and save it:
   ```bash
   python model.py
3. Start the API:
   Run the app.py to start the Flask API:
   ```bash
   python app.py
  The API will be available at http://127.0.0.1:5000/predict.

## File Structure
```graphql
credit-card-fraud-detection/
│
├── data/
│   ├── creditcard.csv          # Original dataset
│   └── processed_data.csv      # Processed dataset
│
├── models/
│   ├── random_forest_model.pkl  # Trained Random Forest model
│   └── scaler.pkl               # StandardScaler object (commented out in preprocess.py)
│
├── preprocess.py                # Data preprocessing script
├── model.py                     # Model training script
└── app.py                       # Flask API for predictions
