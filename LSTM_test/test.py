
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import colors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SpatialDropout1D
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split

def LSTM_Structure (sequence_length, dropout_rate, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test):
    model = Sequential()
    model.add(SpatialDropout1D(dropout_rate, input_shape = (sequence_length, 1)))
    model.add(LSTM(1, dropout = dropout_rate, recurrent_dropout = dropout_rate))
    #model.add(Dense(256, activation = 'relu'))
    #model.add(Dense(128, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    
    opt = optimizers.Adam(lr = learning_rate)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    model.summary()
    
    history = model.fit(x_train, y_train.reshape(-1,num_classes), 
                        epochs = epochs, batch_size = batch_size, verbose = 2, 
                        validation_data = (x_valid, y_valid))
    predicted = model.predict(x_test, verbose = 0)
    predicted = np.argmax(predicted,axis = -1).reshape(-1)

    results = model.evaluate(x_test, y_test,verbose = 0)
    return history, predicted, results

# parameter
Rs = 5
re = 5

dataset_num = 1000

sequence_length = 5
output_dim = 1
num_classes = 2

learning_rate = 0.01
batch_size = 16
epochs  = 200
dropout_rate = 0.3

# dataset
row_input = np.loadtxt('input.csv', delimiter=',',dtype = np.float32)
row_output = np.loadtxt('output.csv', delimiter=',',dtype = np.float32)

x, x_test, y, y_test = train_test_split(row_input, row_output, test_size=0.2, shuffle=False, random_state=1004)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=1004)

y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)
y_valid = to_categorical(y_valid,2)

x_train = x_train.reshape([-1,5,1])
x_test = x_test.reshape([-1,5,1])
x_valid = x_valid.reshape([-1,5,1])

'''
print ("x_train :", np.shape(x_train))
print ("y_train :", np.shape(y_train))
print ("x_test :", np.shape(x_test))
print ("y_test :", np.shape(y_test))
print ("x_valid :", np.shape(x_valid))
print ("y_valid :", np.shape(y_valid))
'''

# LSTM
history, predicted, results = LSTM_Structure (sequence_length, dropout_rate, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test)

# accuracy & loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epo = range(1,len(acc)+1)
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(2, figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(epo, acc, 'bo', label='Training acc')
plt.plot(epo, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs') 
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(epo, loss, 'bo',label = 'Training loss')
plt.plot(epo, val_loss, 'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.savefig('Training and Validation.png', bbox_inches = 'tight')
plt.legend()
plt.show()
