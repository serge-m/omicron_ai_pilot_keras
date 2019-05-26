#!/usr/bin/python3

from keras.models import load_model
from keras_preprocessing.image.utils import load_img, img_to_array

import pandas as pd
import numpy as np
import os

ROW_TO_PREDICT=300

df = pd.read_csv('../data_set/data.csv', header=None, names=['x_col', 'y_col', 'z_col'])[['x_col', 'y_col']]

model = load_model('best_model.h5')
path=os.path.join('../data_set/', df.iloc[ROW_TO_PREDICT]['x_col'])
print(path)
image = img_to_array(load_img(path, target_size=(32,32)))
image=np.expand_dims(image,0)
prediction = model.predict(image)
print(prediction)
