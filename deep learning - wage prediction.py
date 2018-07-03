# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:36:42 2018

@author: MARINAKH

deep learning - wage prediction 
"""

import numpy as np
from urllib.request import urlretrieve
import pandas as pd

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping 
import matplotlib.pyplot as plt

# Assign url of file:url
url = 'https://assets.datacamp.com/production/course_1975/datasets/hourly_wages.csv'

# Save file locally
urlretrieve(url, 'hourly_wage.csv')

#read file into df:
df = pd.read_csv('hourly_wage.csv', sep=',')
df.head()
df.shape
df.describe()
df.info()

# convert to numpy array:
target= df['wage_per_hour'].values
predictors = df.drop(['wage_per_hour'], axis=1).values

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, epochs=30, callbacks=[early_stopping_monitor], validation_split=0.3)
model_0_training = model.fit(predictors, target, epochs=30, callbacks=[early_stopping_monitor], validation_split=0.3, verbose=False)


# Model Summary
model.summary()
model.layers
model.get_config()
model.get_weights()


"""Experimenting with wider networks"""

# Set up the model: model
model_1 = Sequential()

# Add the first layer
model_1.add(Dense(20, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model_1.add(Dense(10, activation='relu'))

# Add the output layer
model_1.add(Dense(1))

# Compile the model
model_1.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model_1.loss)

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model_1_training = model_1.fit(predictors, target, epochs=30, callbacks=[early_stopping_monitor], validation_split=0.3, verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_0_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

# Model Summary
model_1.summary()
model_1.layers
model_1.get_config()
model_1.get_weights()


""" Adding layers to a network """
# Set up the model: model
model_2 = Sequential()

# Add the first layer
model_2.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model_2.add(Dense(32, activation='relu'))
model_2.add(Dense(32, activation='relu'))
model_2.add(Dense(16, activation='relu'))
model_2.add(Dense(1))

# Compile the model
model_2.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model_1.loss)

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model_2.fit(predictors, target, epochs=30, callbacks=[early_stopping_monitor], validation_split=0.3)
model_2_training = model_2.fit(predictors, target, epochs=30, callbacks=[early_stopping_monitor], validation_split=0.3, verbose=False)

# Create the plot
plt.plot(model_2_training.history['val_loss'], 'r', model_0_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
