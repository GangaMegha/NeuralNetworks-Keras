import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

Train_data = pd.read_csv("data/train.csv") #You need to change : create a csv or tsv file of the required format; Assumed to have both train and validation data
Test_data = pd.read_csv("data/test.csv")

# Changing pandas dataframe to numpy array
X = Train_data.iloc[:,:-1].values
y = Train_data.iloc[:,-2:-1].values #You need to change : Use if labels correspond to the last column

X_test = Test_data.iloc[:,:-1].values
y_test = Test_data.iloc[:,-2:-1].values

# Normalizing the data
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test) #Normalise test data based on train data

# Create One-Hot encoded labels
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
y_test = ohe.transform(y_test).toarray()

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) # 80:20 split

# Neural network : Create model
model = Sequential()
model.add(Dense(16, input_dim=20, activation=’relu’))
model.add(Dense(12, activation=’relu’))
model.add(Dense(4, activation=’softmax’))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,validation_data = (X_val,y_val), epochs=100, batch_size=64)

# Prediction on test data
y_pred = model.predict(X_test)

#Converting predictions to label
pred = np.argmax(y_pred, 1) # Check if column is axis 0 or 1

#Converting one hot encoded test label to label
val_test = np.argmax(y_test, 1)

# Metric : Accuracy
a = accuracy_score(pred, val_test)
print('Accuracy is : ', a*100)

# Plot training and validation accuracy v/s epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training and validation loss v/s epoch
plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()