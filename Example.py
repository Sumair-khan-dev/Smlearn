# smlearn Examples
# Author: Sumair Khan
# Version: 0.0.1
# Demonstrates usage of LinearRegression and PolynomailRegression classes

---

## Linear Regression Example (list/array input)

from smlearn import LinearRegression

# Example dataset
X = [[1], [2], [3], [4], [5]]
Y = [2, 4, 6, 8, 10]

# Create and fit the model
model = LinearRegression(X, Y)
model.fit()

# Make predictions
predictions = model.predict([[6], [7]])
print("Linear Regression Predictions:", predictions)
# Expected Output: [12. 14.]

---

## Linear Regression Example (DataFrame input)

import pandas as pd
from smlearn import LinearRegression

# Example DataFrame
data = pd.DataFrame({
    'Feature': [1, 2, 3, 4, 5],
    'Target': [2, 4, 6, 8, 10]  # Target Y must be the last column
})

# The class automatically separates features and target
model = LinearRegression(data)
model.fit()

predictions = model.predict([[6], [7]])
print("Predictions from DataFrame input:", predictions)
# Expected Output: [12. 14.]

---

## Polynomial Regression Example

from smlearn import PolynomailRegression

# Create and fit a polynomial model (degree 2)
poly_model = PolynomailRegression(degree=2, X=X, Y=Y)
poly_model.fit()

# Make predictions
poly_predictions = poly_model.predict([[6], [7]])
print("Polynomial Regression Predictions:", poly_predictions)
# Expected Output: close to [12. 14.] (depending on data and degree)

---

# Notes:
# - Replace X and Y with your own dataset as needed.
# - For multi-feature datasets, provide X as a 2D list/array.
# - Use PolynomailRegression when data is nonlinear.
# - DataFrame input is supported; ensure the target column is last.