#!/usr/bin/env python3
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import autograd.numpy as np
import time

# Neural Network Generator
def generatePrediction(myODE):

    get_custom_objects().update({'custom_activation': Activation(myODE.custom_activation)})

    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim = 1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='custom_activation'))

    #model.output_shape   #model.summary()  #model.get_config()  #model.get_weights()

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(myODE.x_train, myODE.y_train, epochs=20, batch_size=2, verbose=0)

    #score = model.evaluate(x_test, y_test, verbose=1)  #print(score)

    # y_pred contains the prediction with the x_test input
    startNN = time.clock()
    y_pred = model.predict(myODE.x_test)
    print("Total time for NN approximation is: ", time.clock() - startNN)
    return y_pred

# Plotting function
def plot(myODE, y_pred):
    tfit = myODE.x_test
    plt.plot(tfit, y_pred, label='soln')
    plt.plot(tfit, myODE.y_test, 'r--', label='analytical soln')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([myODE.x_test[0], myODE.x_test[len(myODE.x_test)-1]])
    plt.savefig('image.png')

    localError = 0
    for i in range(0, len(y_pred)):
        if myODE.y_test[i] == 0:
            continue
        localError += abs((y_pred[i] - myODE.y_test[i]) /myODE.y_test[i]) * 100
    avgError = localError / len(y_pred)
    print("On average, the error is: ", avgError, " %")
