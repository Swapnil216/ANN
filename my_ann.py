#changing working directory
import os
os.getcwd()
#os.chdir('/Applications/Programming/ML/ANN')

import pandas as pd
#url = 'https://raw.githubusercontent.com/Swapnil216/Umy_ANN/master/Churn_Modelling.csv?token=ANSIXWD3IUKB2P5Y7G5TKVC6VO6RG'
#dataset = pd.read_csv(url)
dataset = pd.read_csv('/Applications/Programming/ML/ML_Udemy/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# labelencoder_X_1 = LabelEncoder()
# X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# onehotencoder = OneHotEncoder(categorical_features = [1])
# X = onehotencoder.fit_transform(X).toarray()

#########
# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
##########
#To avoid dummy variable trap, remove the 0th col.
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential()

#Adding First Hidden Layers
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu',input_dim = 11))

#Adding Second Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid'))

#To clear the model
keras.backend.clear_session()

#Compiling the ANN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# Fitting ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
