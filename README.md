# Credit Card Fraud Detection

## Overview
This project implements a Credit Card Fraud Detection system using machine learning techniques. The system identifies fraudulent transactions by analyzing transaction data and applying a Random Forest classifier. It uses techniques like outlier detection and class balancing to improve the model's performance.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)

## Features
- **Outlier Detection:** Identifies and removes outliers using Isolation Forest.
- **Feature Selection:** Utilizes Random Forest to determine and select important features.
- **Class Balancing:** Implements SMOTE to balance the dataset for improved model performance.
- **Model Training:** Trains a Random Forest classifier and uses Grid Search for hyperparameter tuning.
- **API:** Provides a Flask-based REST API for making predictions.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (for SMOTE)
- Flask
- Matplotlib and Seaborn (for visualizations)

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud) from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders, where the positive class (fraudulent transactions) is a minority class.

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
