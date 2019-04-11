# -*- coding: utf-8 -*-
# TODO:
# test
# refactor
# path handling
# add CLI(docopt)

import os
import csv
import glob
import random
import numpy as np
from PIL import Image
from docopt import docopt

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

##############################################################################
# config
##############################################################################

# PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

# TRAINING
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8


##############################################################################
# keras
##############################################################################

def default_linear():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in

    # Convolution2D class name is an alias for Conv2D
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model


class KerasPilot:
    def __init__(self, model=None, num_outputs=None):
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = default_linear()
        else:
            self.model = default_linear()

    def load(self, model_path):
        self.model = load_model(model_path)

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)
        return hist

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

##############################################################################
# utils
##############################################################################
def expand_path_mask(path):
    matches = []
    path = os.path.expanduser(path)
    for file in glob.glob(path):
        if os.path.isdir(file):
            matches.append(os.path.join(os.path.abspath(file)))
    return matches


def expand_path_arg(path_str):
    path_list = path_str.split(",")
    expanded_paths = []
    for path in path_list:
        paths = expand_path_mask(path)
        expanded_paths += paths
    return expanded_paths
    
def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr

##############################################################################
# data storage
##############################################################################
class DataSetGenerator(object):
    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.data_set = []
        self.train = []
        self.eval = []
        pass

    def read(self):
        with open(self.path, 'r') as csvfile:
            table = csv.reader(csvfile)
            for row in table:
                picture_file_name = row[0]
                # TODO: verify / expand path relative to csv
                pic = Image.open("foo.jpg")
                in_values = np.array(pic)

                out_values = np.array([float(i) for i in row[1:2]])
                self.data_set.append((in_values, out_values))

    def populate_train_eval(self, train_frac=0.6, record_transform=None):
        self.train = random.sample(self.data_set, int(train_frac * len(self.data_set)))
        for i in self.data_set:
            if i not in self.train:
                self.eval.append(i)
                
        if record_transform:
            self.train = record_transform(self.train)
            self.eval = record_transform(self.eval)

    def get_train_generator(self):
        for i in self.train:
            yield i

    def get_eval_generator(self):
        for i in self.eval:
            yield i
        pass


def train(cfg, tub_names, model_name, base_model_path=None):
    """
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    """

    def rt(record):
        record['user/angle'] = linear_bin(record['user/angle'])
        return record

    model_path = os.path.expanduser(model_name)

    kl = KerasPilot()
    if base_model_path is not None:
        base_model_path = os.path.expanduser(base_model_path)
        kl.load(base_model_path)

    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')

    # XXX: what happens if total_train is not multiple of BATCH_SIZE? 
    data_set = DataSetGenerator(tub_names)
    data_set.read()
    data_set.populate_train_eval(record_transform=rt,
                                 train_frac=TRAIN_TEST_SPLIT)
    train_gen = data_set.get_train_gen()
    val_gen = data_set.get_eval_gen()

    total_records = len(data_set.data_set)
    total_train = int(total_records * TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    steps_per_epoch = total_train // BATCH_SIZE

    print('tub_names', tub_names)
    print('train: %d, validation: %d' % (total_train, total_val))
    print('steps_per_epoch', steps_per_epoch)

    kl.train(train_gen,
             val_gen,
             saved_model_path=model_path,
             steps=steps_per_epoch,
             train_split=cfg.TRAIN_TEST_SPLIT)
