# Multiple linear regression

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Categorical data
encoder = OneHotEncoder()
encoded_X = np.array(encoder.fit_transform(X[:, [3]]).toarray())
X = np.concatenate((X, encoded_X), axis=1)
X = np.delete(X, 3, axis=1)

# Avoiding the dummy variable trap by deleting one dummy variable
X = X[:, :-1]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Fitting Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the result
y_pred = regressor.predict(X_test)
