# Simple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting results
y_pred = regressor.predict(X_test)
accuracy = regressor.score(X_test, y_test)

# Visualising the training set results
plt.scatter(X_train, y_train, color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.title('Salary and Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the test set results
plt.scatter(X_test, y_test, color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.title('Salary and Experience (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
