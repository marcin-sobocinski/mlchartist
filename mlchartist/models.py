
from tensorflow.keras.metrics import Precision, BinaryAccuracy
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras import layers, models 
from tensorflow.keras.optimizers import RMSprop, Adam, Adamax

optim = RMSprop(learning_rate=0.0001)
precision = Precision(thresholds=0.8)
bin_accuracy = BinaryAccuracy(threshold=0.8)

def final_model():
    model = Sequential()
    reg_l1 = regularizers.l1(0.001)
    reg_l2 = regularizers.l2(0.001)
    reg_l1_l2 = regularizers.l1_l2(l1=0.001, l2=0.001)
    model.add(layers.LSTM(300, return_sequences=True, input_shape=(30,13), activation='tanh'))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(300, activation='tanh'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(300, activation='relu', kernel_regularizer=reg_l1))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(200, activation='relu', bias_regularizer=reg_l2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(100, activation='relu', activity_regularizer=reg_l1_l2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[precision, bin_accuracy])
    
    return model

def simple_model():
    model = Sequential()
    model.add(layers.LSTM(units=10,  activation='tanh')) 
    model.add(layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=optim, metrics=['accuracy'])
    return model
