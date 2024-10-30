# preprocess.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("data/creditcard.csv")
print(df.columns)
print(df.info())
print(df.isnull().sum())

#Drop the target variable and scale features
features = df.drop(columns='Class')
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# # Apply Isolation Forest for outlier detection
# iso = IsolationForest(contamination=0.1, random_state=1)
# outliers = iso.fit_predict(features_scaled)
# df = df[outliers != -1]  # Keep only non-outliers
# features = df.drop(columns='Class')
y = df['Class']

# # Re-scaling after outlier removal
# features_scaled = scaler.fit_transform(features)

#Feature selection using RandomForest
rf = RandomForestClassifier(random_state=1)
rf.fit(features_scaled, y)
feature_importance = rf.feature_importances_

# Select top features
top_features = [feature for feature, importance in zip(features.columns, feature_importance) if importance > 0.05]
df_selected = df[top_features + ['Class']]  # Keep top features and the target variable

# Save processed data and scaler
df_selected.to_csv("data/processed_data.csv", index=False)
# joblib.dump(scaler, 'models/scaler.pkl')
print("Data preprocessing complete and saved to 'data/processed_data.csv'")
print("Scaler saved to 'models/scaler.pkl'")
