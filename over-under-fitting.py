#reducing the network's size
#original model
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#version of model with lower capacity
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#version of the model with higher capacity
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#adding weight regularization
#Adding L2 weight regularization to the model
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#different weight regularizers available in Keras
from keras import regularizers
regularizers.l1(0.001) #l1 regularization
regularizers.l1_l2(l1=0.001, l2=0.001) # simultaneousion L1 and L2 regularization

#Adding dropout
'''
Dropout is one of the most effective and most commonly used regularization tech-
niques for neural networks, developed by Geoff Hinton and his students at the Uni-
versity of Toronto.'''

#adding dropout to IMDB network
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

'''
To recap, these are the most common ways to prevent overfitting in neural networks:
 Get more training data.
 Reduce the capacity of the network.
 Add weight regularization.
 Add dropout.'''