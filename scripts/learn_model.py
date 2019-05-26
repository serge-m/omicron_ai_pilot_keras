#!/usr/bin/python3

from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers

import pandas as pd

df = pd.read_csv('../data_set/data.csv', header=None, names=['x_col', 'y_col', 'z_col'])[['x_col', 'y_col']]
df['y_col'] = ((df['y_col'] * 2).astype('int').astype('float')/2).astype('str')
print(df)

datagen=ImageDataGenerator()
train_generator=datagen.flow_from_dataframe(dataframe=df, directory="../data_set", x_col="x_col", y_col="y_col", class_mode="categorical", target_size=(32,32), batch_size=32)
model = Sequential()

model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(Convolution2D(32, (5, 5), strides=(2,2), activation='relu', padding='same'))
model.add(Convolution2D(64, (5, 5), strides=(2,2), activation='relu', padding='same'))
model.add(Convolution2D(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
model.add(Convolution2D(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(100, activation='linear'))
model.add(Dropout(rate=.1))
model.add(Dense(50, activation='linear'))
model.add(Dropout(rate=.1))
model.add(Dense(5, activation='linear'))

model.compile(optimizers.rmsprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='acc', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='acc', save_best_only=True)]

model.fit_generator(callbacks=callbacks, generator=train_generator, steps_per_epoch=10, epochs=10)
