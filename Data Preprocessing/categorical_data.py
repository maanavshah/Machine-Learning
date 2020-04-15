# Categorical Data

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Categorical data
# Independant variable
ct = ColumnTransformer(
    [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
# Dependant variable
y = LabelEncoder().fit_transform(y)
