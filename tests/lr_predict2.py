import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression as sklr

#from linear_regression import LinearRegression
from ren_algorithm import LinearRegression


""" 
    multi linear regression
"""


# Generate synthetic data with multiple features
X, y = datasets.make_regression(n_samples=100, n_features=3, noise=20, random_state=4)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# Initialize the Multiple Linear Regression model and train
reg = LinearRegression(lr=0.01, iterations=1000)
reg.learn(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)

# Define Mean Squared Error (MSE) function
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Evaluate the model
mse = calculate_mse(y_test, predictions)
# Calculate RMSE
rmse = np.sqrt(mse)
# Calculate R² (R-squared)
r2 = r2_score(y_test, predictions)


# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse_value = calculate_mse(y_test, predictions)
    rmse_value = np.sqrt(mse_value)
    r2_value = r2_score(y_test, predictions)
    
    return mse_value, rmse_value, r2_value

sk_model = sklr()
sk_mse, sk_rmse, sk_r2 = evaluate_model(sk_model, X_train, X_test, y_train, y_test)

results = [
    ["Built-in Linear Regression", sk_mse, sk_rmse, sk_r2],
    ["Custom Multiple Linear Regression", mse, rmse, r2]
]

# Create a DataFrame
df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R²"])
print("\n",df,"\n")
