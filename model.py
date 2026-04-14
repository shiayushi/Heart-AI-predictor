# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset (Heart Disease UCI format)
df = pd.read_csv("heart.csv")

# Features & target
X = df.drop("target", axis=1)
y = df["target"]

# Save feature names (IMPORTANT)
feature_names = X.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Save everything
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("features.pkl", "wb") as f:
    pickle.dump(feature_names, f)
    
import shap
# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# SHAP Explainer
explainer = shap.TreeExplainer(model)
pickle.dump(explainer, open("explainer.pkl", "wb"))

print("✅ Model + SHAP saved successfully!")