import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# read data
bodyswing_df = pd.read_csv("OneHand.txt")
handswing_df = pd.read_csv("TwoHands.txt")

X = []
Y = []
nb_of_timesteps = 10


dataset = bodyswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(nb_of_timesteps, n_sample):
    X.append(dataset[i - nb_of_timesteps:i, :])
    Y.append(1)

dataset = handswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(nb_of_timesteps, n_sample):
    X.append(dataset[i - nb_of_timesteps:i,:])
    Y.append(0)

X, Y = np.array(X), np.array(Y)
# print(X.shape, Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

model = Sequential()
# model.add(LSTM(units = 128, input_shape = (nb_of_timesteps, X.shape[2])))
# model.add(Dense(units="number of class", activation="softmax" ))
# model.add(Dense(units = 1 , activation= "sigmoi"))
model.add(LSTM(units = 50, return_sequences = True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation = "sigmoid"))
model.compile(optimizer="adam", metrics=['accuracy'], loss= "binary_crossentropy")

model.fit(X_train, Y_train, epochs=16, batch_size=32, validation_data= (X_test, Y_test))
model.save("HumanModel1.h5")

