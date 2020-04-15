# Artifical Neural Network (ANN)

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, [-1]].values

# Encoding categorical data
X[:, 1] = LabelEncoder().fit_transform(X[:, 1])
ct = ColumnTransformer(
    [("encoder", OneHotEncoder(), [2])], remainder="passthrough")
X = ct.fit_transform(X)
X = np.delete(X, [0], axis=1)  # Avoiding dummy variable trap

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
# output_dim (nodes) => average of input features + ouput features
classifier.add(Dense(output_dim=6, init='uniform',
                     activation='relu', input_dim=10))

# Adding second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the results
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
