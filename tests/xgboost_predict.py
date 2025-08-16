import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#from xgboost_regression import CustomXGBoostRegressor
from ren_algorithm import CustomXGBoostRegressor


# Load the California housing dataset
data = fetch_california_housing()

# Use only a subset of the dataset (e.g., 2000 samples)
X = data.data[:700]
y = data.target[:700]

print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
#X, y = data.data, data.target  # Features and target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor
xgb_reg = CustomXGBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model
xgb_reg.fit(X_train, y_train)

# Make predictions
predictions = xgb_reg.predict(X_test)

# Evaluate model performance
mse = mse(y_test, predictions)
rmse = np.sqrt(mse)

print(f"actually price estimation: {y_test[:5]}")
print(f"Predictions: {predictions[:5]}")  # Print first 5 predictions
print(f"RMSE: {rmse}") 

