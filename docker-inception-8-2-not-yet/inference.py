
# coding: utf-8
import os.path as osp
import openslide
from pathlib import Path
# https://devstorylog.blogspot.com/2018/05/anaconda-python-vscode.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from skimage.filters import threshold_otsu
from openslide.deepzoom import DeepZoomGenerator
import cv2
from keras.utils.np_utils import to_categorical

# network
from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

# Unet
import numpy as np 
import os

import skimage.transform as trans
#import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

# train
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

# evaluate
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import math
from PIL import Image
from xml.etree.ElementTree import ElementTree, Element, SubElement
from io import BytesIO
import skimage.io as io

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import keras.backend.tensorflow_backend as K
from sklearn import metrics
from keras.preprocessing.image import *
from model import find_patches_from_slide, gen_imgs, predict_batch_from_model, predict_from_model, InceptionV3, unet, simple_model, gen_imgs_test
print('****************************INFERENCE FILE*******************************')
#model = simple_model(pretrained_weights ='/data/model/u_1.h5')
model = InceptionV3(include_top=True,
                weights='/data/model/i_1.h5',
                input_tensor=None,
                input_shape=(256,256,3),
                pooling=None,
                classes=2)

PATCH_SIZE = 256
NUM_CLASSES = 2 # not_tumor, tumor

file_handles=[]

def read_test_data_path_2():
    path_dir = '/data/test/'
    file_list = os.listdir(path_dir)
    file_list.sort()
    paths = []
    for pt in file_list:
        if 'mrxs' in pt:
            paths.append(path_dir + pt)
    return paths


PATCH_SIZE = 256
BATCH_SIZE = 128
test_image_paths = read_test_data_path_2()
print(test_image_paths)

start_x = 64
start_y = 64
pred_size = 128

slide_id = list()
slide_pred = list()
for id_test in range(len(test_image_paths)):
    print(id_test,'th inference\n')
    image_path = test_image_paths[id_test]
    test_samples = find_patches_from_slide(image_path,'test')
    NUM_SAMPLES = len(test_samples)

    if NUM_SAMPLES > 7000:
        NUM_SAMPLES = 7000

    samples = test_samples.sample(NUM_SAMPLES, random_state=42)
    samples.reset_index(drop=True, inplace=True)

    test_generator = gen_imgs_test(image_path, 'test', samples, BATCH_SIZE)
    test_steps = np.ceil(len(samples)/BATCH_SIZE)


    preds = []
    for i in range(int(test_steps)):
        X, Y = next(test_generator)
        for j in range(len(X)):
            prediction = predict_from_model(X[j],model)
            pred_X = np.zeros((pred_size,pred_size))
            for x in range(start_x,start_x+pred_size):
                for y in range(start_y, start_y+pred_size):
                    pred_X[x-start_x][y-start_y] = prediction[x][y]
            pred_s = pd.Series(pred_X.flatten())
            pre_p = np.sort(pred_s)[7272]
            pred_x_i = pre_p
            preds.append(pred_x_i)
        
    max_pred_x = np.max(preds)
    print(id_test,"'s max pred : ",max_pred_x)
    slide_id.append(test_image_paths[id_test][11:19])
    slide_pred.append(max_pred_x)
    for fh in file_handles:
        fh.close()
    file_handles = []

# csv file 만들기
# list로 만든다음에 넣기 
# okay==
df = pd.DataFrame()
df['slide_id'] = slide_id
df['slide_pred'] = slide_pred
path = '/data/output'
df.to_csv(path+'/output.csv', index=False, header=False)
print('test df file completed')


