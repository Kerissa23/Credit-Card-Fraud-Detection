# model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,classification_report, roc_auc_score
import joblib

# Load preprocessed data
df_selected = pd.read_csv("data/processed_data.csv")

# Split data into features and target
X = df_selected.drop(columns='Class')
y = df_selected['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Define the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Define parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Set up Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1)

# Fit the model
grid_search.fit(X_train_sm, y_train_sm)

# Predict and evaluate
y_pred = grid_search.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]))
print("Accuracy Score:",accuracy_score(y_test,y_pred))
# Save the model
joblib.dump(grid_search.best_estimator_, "models/random_forest_model.pkl")
print("Model saved to 'models/random_forest_model.pkl'")
