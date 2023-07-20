import numpy as np
import pandas as pd


data_dir='/kaggle/input/ucihar-dataset/UCI-HAR Dataset'
# Raw data signals
#Signals are from Accelerometer and gyroscope
#The signals are in x,y,z directions
#sensor signals are filtered to have only body acceleration
# excluding the acceleration due to gravity
# Triaxial acceleration from the accelerometer is total acceleration

SIGNALS=[
    "body_acc_x",
    "body_acc_x",
    "body_acc_x",
    "body_gyro_x",
    "body_gyro_x",
    "body_gyro_x",
    "total_acc_x",
    "total_acc_x",
    "total_acc_x",
    
]


## Utility function to read the data from csv file
def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

#Utility functions to load the signals
def load_signals(subset):
    signals_data=[]
    for signal in SIGNALS:
        filename=f'/kaggle/input/ucihar-dataset/UCI-HAR Dataset/{subset}/Inertial Signals/{signal}_{subset}.txt'    #provide the directory here
        signals_data.append(
        _read_csv(filename).values
        )
        
    # Transpose is used to change the dimesionality of the output,
    # Aggregating the signals by combination of sample/timestep.
    # Resulatant shape is (7352 train/ 2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data,(1,2,0))

def load_y(subset):
    """the objective that we are trying to predict is a integer from 1 to 6, that represents 
    a human activity representation of every sample objective as a 6 bits vector using One Hot Encoding"""
    
    filename=f'/kaggle/input/ucihar-dataset/UCI-HAR Dataset/{subset}/y_{subset}.txt'                           #'directory of inertial signal'
    y=_read_csv(filename)[0]
    return pd.get_dummies(y).values

def load_data():
    """ obtain the dataset from multiple files 
    Returns: x_train, x_test, y_train, y_test"""
    x_train, x_test=load_signals('train'), load_signals('test')
    y_train,y_test=load_y('train'),load_y('test')
    
    return x_train, x_test,y_train,y_test

# Importing tensorflow
import numpy as np
np.random.seed(42)
import tensorflow as tf
#tf.set_random_seed(42)

# import keras
from keras import backend as k
# Create a TensorFlow session
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())

# Set the session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)


# configuring a session
session_conf=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

# initializing Parameters
epochs=30,
batch_size=16
n_hidden=32

# to count the number of classes
def _count_classes(y):
    return len(set([tuple(category) for category in y]))

# Loading the train and test data
x_train, x_test, y_train, y_test= load_data()


timesteps=len(x_train[0])
input_dim=len(x_train[0][0])
n_classes= _count_classes(y_train)

print(timesteps)
print(input_dim)
print(len(x_train))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

# Initializing the sequential model
model=Sequential()
model.add(LSTM(n_hidden, input_shape=(timesteps,input_dim)))
model.add(Dropout(0.5))    ## here we are using dropout to overcome the overfiting because the no. of paramteres
                           ## are large 5574 and the observations are less 10000(train+test)
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()


# this is simple LSTM we can add layers to experiment 

# Compiling Model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train,
         batch_size=batch_size,
         validation_data=(x_test, y_test),
         epochs=epochs[0])

score=model.evaluate(x_test, y_test)
score

"""93/93 [==============================] - 1s 14ms/step - loss: 0.4911 - accuracy: 0.8208
[0.4911234676837921, 0.820834755897522]"""
