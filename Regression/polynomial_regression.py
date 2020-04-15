# Polynomial Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, -1].values

# Fitting Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial Regression
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Visualising the Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(poly_regressor.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(
    poly_regressor.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising Linear Regression vs Polynomial Regression
plt.scatter(X, y, color="red")
plt.plot(X, linear_regressor.predict(X), color="blue")
plt.plot(X, regressor.predict(poly_regressor.fit_transform(X)), color="green")
plt.title("Linear Regression vs. Polynamial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot()
