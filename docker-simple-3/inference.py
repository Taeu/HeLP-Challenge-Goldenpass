
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


from train import find_patches_from_slide, read_test_data_path, predict_from_model, simple_model, unet

#model = simple_model(pretrained_weights ='/data/model/u_1.h5')
model = simple_model(pretrained_weights = 's_1.h5')

PATCH_SIZE = 256
NUM_CLASSES = 2 # not_tumor, tumor

def gen_imgs_test(slide_path, truth_path, samples, batch_size, patch_size = PATCH_SIZE,num_epoch = 1, shuffle=True):
    """This function returns a generator that 
    yields tuples of (
        X: tensor, float - [batch_size, patch_size, patch_size, 3]
        y: tensor, int32 - [batch_size, patch_size, patch_size, NUM_CLASSES]
    )
    
    
    input: samples: samples dataframe
    input: batch_size: The number of images to return for each pull
    output: yield (X_train, y_train): generator of X, y tensors
    
    option: base_truth_dir: path, directory of truth slides
    option: shuffle: bool, if True shuffle samples
    """
    
    num_samples = len(samples)
    slide_contains_tumor = 'pos' in slide_path    
    slide = openslide.open_slide(slide_path)
    #with openslide.open_slide(slide_path) as slide:
    tiles = DeepZoomGenerator(slide,tile_size=patch_size, overlap=0, limit_bounds=False) # 이거 limit_bounds =True하면 저거 굳이 안가져와도 될텐데
    
    start_x = int(slide.properties.get('openslide.bounds-x',0))
    start_y = int(slide.properties.get('openslide.bounds-y',0))
    start_x = start_x / patch_size
    start_y = start_y / patch_size

        #img = tiles.get_tile(tiles.level_count-1, (x,y))
    
    
    if slide_contains_tumor:
        truth = openslide.open_slide(truth_path)
        #with openslide.open_slide(truth_path) as truth:
        truth_tiles = DeepZoomGenerator(truth, tile_size=16, overlap=0, limit_bounds=False)
            
    
    
    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1) # shuffle samples

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            masks = []
            for _, batch_sample in batch_samples.iterrows(): # 배치마다 deep zoom 하네 약간 비효율적
                
                x, y = batch_sample.tile_loc[::-1]
                x += start_x
                y += start_y
                img = tiles.get_tile(tiles.level_count-1, (x,y))
                
                mask_size_up = np.zeros((patch_size,patch_size))
                a,b=mask_size_up.shape
                # only load truth mask for tumor slides
                if slide_contains_tumor:
                    mask = truth_tiles.get_tile(truth_tiles.level_count-1, batch_sample.tile_loc[::-1])
                    mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        # mask_size_up , 16 to 256
                    k, l = mask.shape
                    for i in range(k):
                        for j in range(l):
                            for o in range(16):
                                for p in range(16):
                                    mask_size_up[i*16+o,j*16+p] = mask[i][j]

                if img.size != (patch_size,patch_size):
                    print('this tisuue shape is not ',(patch_size,patch_size))
                    img = Image.new('RGB', (patch_size,patch_size))
                    mask_size_up = np.zeros((patch_size,patch_size))


                images.append(np.array(img))
                masks.append(mask_size_up)

            
            X_train = np.array(images)
            y_train = np.array(masks)
            #print('x_train_shape :', X_train.shape)
            
            y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], patch_size, patch_size, 2) 
            #print('y_train_shape : ',y_train.shape)
            yield X_train, y_train


PATCH_SIZE = 256
BATCH_SIZE = 32
test_image_paths = read_test_data_path()

start_x = PATCH_SIZE//4
start_y = PATCH_SIZE//4
pred_size = PATCH_SIZE//2

slide_id = list()
slide_pred = list()
for i in range(len(test_image_paths)):
    print(i,'th inference\n')
    image_path = test_image_paths[i]
    test_samples = find_patches_from_slide(image_path,'test')
    NUM_SAMPLES = len(test_samples)
    if NUM_SAMPLES > 1000:
        NUM_SAMPLES = 1000

    samples = test_samples.sample(NUM_SAMPLES, random_state=42)
    samples.reset_index(drop=True, inplace=True)

    test_generator = gen_imgs_test(image_path, 'test', samples, BATCH_SIZE)
    test_steps = np.ceil(len(samples)/BATCH_SIZE)


    preds = []
    for i in tqdm(range(int(test_steps))):
        X, Y = next(test_generator)
        for j in range(len(X)):
            prediction = predict_from_model(X[j],model)
            pred_X = np.zeros((pred_size,pred_size))
            for x in range(start_x,start_x+pred_size):
                for y in range(start_y, start_y+pred_size):
                    pred_X[x-start_x][y-start_y] = prediction[x][y]
            pred_s = pd.Series(pred_X.flatten())
            pred_x_i = np.max(pred_s)
            preds.append(pred_x_i)
    max_pred_x = np.max(preds)
    
    slide_id.append(test_image_paths[i][11:19])
    slide_pred.append(max_pred_x)
# csv file 만들기
# list로 만든다음에 넣기 
# okay==
df = pd.DataFrame()
df['slide_id'] = slide_id
df['slide_pred'] = slide_pred
path = '/data/output'
df.to_csv(path+'/output.csv', index=False, header=False)
print('test df file completed')


