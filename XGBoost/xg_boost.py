# XGBoost

# Installing XGBoost
# Anaconda Users: conda install -c conda-forge xgboost

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, [-1]].values

# Categorical data
ct = ColumnTransformer(
    [("encoder", OneHotEncoder(), [1, 2])], remainder="passthrough")
X = ct.fit_transform(X)
X = np.delete(X, [0, 3], axis=1)

# Splitting the datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Fitting XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
