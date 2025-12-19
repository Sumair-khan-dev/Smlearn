"""
Graphing examples for smlearn
Visualizes LinearRegression and PolynomialRegression fits
"""

import numpy as np
import matplotlib.pyplot as plt
from smlearn import LinearRegression, PolynomailRegression

# =========================
# LINEAR REGRESSION DATA
# =========================
X_linear = [[1],[2],[3],[4],[5]]
Y_linear = [7, 9, 11, 13, 15]

lin_model = LinearRegression(X=X_linear, Y=Y_linear)
lin_model.fit()

# Predictions for plotting
X_plot = np.linspace(1, 5, 100).reshape(-1,1)
Y_plot = lin_model.predict(X_plot)

# =========================
# POLYNOMIAL REGRESSION DATA
# =========================
X_poly = [[1],[2],[3],[4],[5]]
Y_poly = [4, 9, 16, 25, 36]

poly_model = PolynomailRegression(degree=2, X=X_poly, Y=Y_poly)
poly_model.fit()

X_plot_poly = np.linspace(1, 5, 100).reshape(-1,1)
Y_plot_poly = poly_model.predict(X_plot_poly)

# =========================
# PLOTTING
# =========================

plt.figure(figsize=(12,5))

# Linear
plt.subplot(1,2,1)
plt.scatter(X_linear, Y_linear, color='blue', label='Data points')
plt.plot(X_plot, Y_plot, color='red', label='Linear fit')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# Polynomial
plt.subplot(1,2,2)
plt.scatter(X_poly, Y_poly, color='green', label='Data points')
plt.plot(X_plot_poly, Y_plot_poly, color='orange', label='Polynomial fit')
plt.title('Polynomial Regression (Degree 2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()