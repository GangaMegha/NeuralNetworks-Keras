import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

def plot_acc(history, args):
	# Plot training and validation accuracy v/s epoch
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(args.save_dir + "/Plot_accuracy_train_val.png")
	plt.show()

def plot_loss(history, args):
	# Plot training and validation loss v/s epoch
	plt.plot(history.history['loss']) 
	plt.plot(history.history['val_loss']) 
	plt.title('Model loss') 
	plt.ylabel('Loss') 
	plt.xlabel('Epoch') 
	plt.legend(['Train', 'Test'], loc='upper left') 
	plt.savefig(args.save_dir + "/Plot_loss_train_val.png")
	plt.show()

def read_data(args):
	Train_data = pd.read_csv(args.train_file, sep=" ") #You need to change : create a csv or tsv file of the required format; Assumed to have both train and validation data
	Test_data = pd.read_csv(args.test_file, sep=" ")

	nc = Train_data.shape[1] #number of columns
	# Changing pandas dataframe to numpy array
	X = Train_data.iloc[:,:nc-1].values
	y = Train_data.iloc[:,nc-1:nc].values #You need to change : Use if labels correspond to the last column

	X_test = Test_data.iloc[:,:nc-1].values
	y_test = Test_data.iloc[:,nc-1:nc].values

	return X, X_test, y, y_test

def get_data(args):
	X, X_test, y, y_test = read_data(args)

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

	return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(input_shape, args):
	# Neural network : Create model
	model = Sequential()
	model.add(Dense(16, input_dim=input_shape, activation=args.activation))
	model.add(Dense(12, activation=args.activation))
	model.add(Dense(2, activation="softmax"))

	return model

def evaluate(X_test, y_test, model):
	# Prediction on test data
	y_pred = model.predict(X_test)

	#Converting predictions to label
	pred = np.argmax(y_pred, 1) # Check if column is axis 0 or 1

	#Converting one hot encoded test label to label
	val_test = np.argmax(y_test, 1)

	# Metric : Accuracy
	a = accuracy_score(pred, val_test)
	print('Accuracy is : ', a*100)

def execute(args):
	X_train, X_val, X_test, y_train, y_val, y_test = get_data(args)

	# Create the model architecture
	model = create_model(X_val.shape[1], args)

	# Compile the model
	model.compile(loss=args.loss, optimizer='adam', metrics=['accuracy'])

	# Train the model
	history = model.fit(X_train, y_train, validation_data = (X_val,y_val), epochs=10, batch_size=args.batch_size)

	# Plot loss
	plot_loss(history, args)

	# Save model and architecture to single file
	model.save(args.save_dir + "/model.h5")

	# Evaluate on test set
	evaluate(X_test, y_test, model)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file', action="store", dest="train_file", default="data/train.txt", type = str)
	parser.add_argument('--test_file', action="store", dest="test_file", default="data/test.txt", type = str)

	parser.add_argument('--lr', action="store", dest="lr", default=1, type = float)
	parser.add_argument('--activation', action="store", dest="activation", default="relu", type = str)
	parser.add_argument('--loss', action="store", dest="loss", default="categorical_crossentropy", type = str)
	parser.add_argument('--batch_size', action="store", dest="batch_size", default=1, type = int)
	parser.add_argument('--epoch', action="store", dest="epoch", default=1, type = int)
	parser.add_argument('--save_dir', action="store", dest="save_dir", default="model_results", type = str)

	args = parser.parse_args()

	# Read the data and train the network
	execute(args)



#--------------------Miscellaneous--------------
# def load_saved_model():
# 	# load and evaluate a saved model
# 	from numpy import loadtxt
# 	from keras.models import load_model
	 
# 	# load model
# 	model = load_model('model.h5')
# 	# summarize model.
# 	model.summary()
# 	# load dataset
# 	dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 	# split into input (X) and output (Y) variables
# 	X = dataset[:,0:8]
# 	Y = dataset[:,8]
# 	# evaluate the model
# 	score = model.evaluate(X, Y, verbose=0)
# 	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))