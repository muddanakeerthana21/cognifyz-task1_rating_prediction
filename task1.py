# ==========================================
# TASK 1 - RESTAURANT RATING PREDICTION
# Cognifyz ML Internship
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------
# Load Dataset
# ------------------------------------------

# Change filename if needed
df = pd.read_csv("Dataset .csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# ------------------------------------------
# Dataset Information
# ------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# ------------------------------------------
# Missing Values
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------
# Select Features and Target
# ------------------------------------------

X = df[
    [
        "Votes",
        "Price range",
        "Average Cost for two"
    ]
]

y = df["Aggregate rating"]

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ------------------------------------------
# Predictions
# ------------------------------------------

predictions = model.predict(X_test)

# ------------------------------------------
# Model Evaluation
# ------------------------------------------

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("----------------------")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

results = pd.DataFrame({
    "Actual Rating": y_test,
    "Predicted Rating": predictions
})

print("\nActual vs Predicted Ratings")
print(results.head(10))

# ------------------------------------------
# Feature Importance
# ------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Influence")
print(coefficients)

# ------------------------------------------
# Visualization
# ------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Restaurant Ratings")

plt.show()

print("\nTask 1 Completed Successfully!")