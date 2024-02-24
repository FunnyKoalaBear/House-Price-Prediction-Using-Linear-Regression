import keras #allows the actual deep learning network to be made
import numpy as np #does the math 
import pandas as pd  #Facilitates data manipulation and analysis through its powerful data structures and functions
import tensorflow as tf #library to create a basic neural network. Input layer -> hidden layer -> Output layer
from sklearn.model_selection import train_test_split #library that splits dataset into a testing set and a dataset, done to test model on data it hasnt seen before
import matplotlib.pyplot as plt #


dataset = pd.read_csv('/Users/sbdagoat/Desktop/Codine/Python/AI/HousePrice 1/IowaHousingPrices.csv')
print(dataset.columns)


#x variable is being set as squareFeet
squareFeet = dataset[["SquareFeet"]].values #the double square bracket allows the column to be selected by name 
#y variable is being set as salePrice
salePrice = dataset[["SalePrice"]].values #.values will use numpy to put values into an array 
print("Square feet")
print(squareFeet)

print("Sale price")
print(salePrice)


model = keras.Sequential() #allows to add layers sequentially to the model


#input layer 
model.add(keras.layers.Dense(1, input_shape=(1, ))) #this adds a dense layer that takes inputs 
#takes 1 input according to input shape 



#compiling the mode, configures how it should be trained 
model.compile(keras.optimizers.Adam(learning_rate=1.0), 'mean_squared_error')
#this reduces the underfitting and overfitting of the linear regression analysis by using an algorithm called Adam
#Learning rate determines much the fitting line will curve, 


#now to start the training process 
model.fit(squareFeet, salePrice, epochs = 30, batch_size = 10) #fitting the x and y into the model
#epochs determines how many times it runs over the data 
#batch size tells you how many data points enter model at a time 



#Plot datapoints
dataset.plot(kind='scatter',
       x='SquareFeet',
       y='SalePrice', title='Housing Prices and Square Footage of Iowa Homes')


y_pred = model.predict(squareFeet) #The predicted housing price based on square feet
#model predicts with square feet that is in dataset and plots it. 

#Plot the linear regression line
plt.plot(squareFeet, y_pred, color='red')


#displays the plot 
plt.show()


#now to predict a new dataset 

newSF = 2000
# Reshape the input to match the model's input shape
newSF_array = np.array([[newSF]])
print(model.predict(newSF_array))
