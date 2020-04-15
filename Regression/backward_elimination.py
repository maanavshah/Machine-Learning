# Backward Elimination using linear regression

# Importing the libraries
import statsmodels.api as sm
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
enc_df = np.array(encoder.fit_transform(X[:, [3]]).toarray())
X = np.concatenate((X, enc_df), axis=1)
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

# Building the optimal model using Backword Elimination
# adding 1s as first column (Intercept)
X = np.append(arr=np.ones((50, 1)), values=X, axis=1).astype(int)

X_optimize = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_optimize).fit()
print(regressor_OLS.summary())

X_optimize = X[:, [0, 1, 2, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_optimize).fit()
print(regressor_OLS.summary())

X_optimize = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_optimize).fit()
print(regressor_OLS.summary())

X_optimize = X[:, [0, 1]]
regressor_OLS = sm.OLS(endog=y, exog=X_optimize).fit()
print(regressor_OLS.summary())

# p-values and Adjusted R Squared:


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# p-values and Adjusted R Squared:


def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


backwardElimination(X, 0.05)
