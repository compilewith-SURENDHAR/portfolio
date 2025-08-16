from statistics import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression as sklr

#from linear_regression import LinearRegression
from ren_algorithm import LinearRegression


""" 
    simple linear regression
"""

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

reg = LinearRegression(lr=0.01)
reg.learn(X_train,y_train)
predictions = reg.predict(X_test)

def calculate_mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)


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


y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()