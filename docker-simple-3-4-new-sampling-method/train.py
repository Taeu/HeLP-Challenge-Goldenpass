
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
#from xml.etree.ElementTree import ElementTree, Element, SubElement
from io import BytesIO
import skimage.io as io

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import keras.backend.tensorflow_backend as K
from sklearn import metrics
from keras.preprocessing.image import *

PATCH_SIZE = 256
IS_TRAIN = True
BATCH_SIZE = 32
N_EPOCHS = 5

NUM_CLASSES = 2 # not_tumor, tumor
file_handles=[]

def gen_imgs(samples, batch_size,patch_size = PATCH_SIZE, shuffle = True):
    
    num_samples = len(samples)
    while 1:
        if shuffle:
            samples = samples.sample(frac=1)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            masks = []
            for _, batch_sample in batch_samples.iterrows():
                slide_path = batch_sample.slide_path
                print(slide_path)
                slide_contains_tumor = 'pos' in slide_path
                mask_size_up = np.zeros((patch_size,patch_size))
                a,b=mask_size_up.shape

                with openslide.open_slide(slide_path) as slide:
                    tiles = DeepZoomGenerator(slide, tile_size=patch_size,overlap=0,limit_bounds=False)
                    if slide_contains_tumor:
                        start_x = int(slide.properties.get('openslide.bounds-x',0))
                        start_y = int(slide.properties.get('openslide.bounds-y',0))
                        start_x = start_x / patch_size
                        start_y = start_y / patch_size

                        x,y = batch_sample.tile_loc[::-1]
                        x += start_x
                        y += start_y

                        img = tiles.get_tile(tiles.level_count-1,(x,y))


                        truth_slide_path = '/data/train/mask/positive/' + slide_path[27:35] + '.png'
                        with openslide.open_slide(truth_slide_path) as truth:
                            truth_tiles = DeepZoomGenerator(truth,tile_size=16, overlap = 0, limit_bounds = False)
                            mask = truth_tiles.get_tile(truth_tiles.level_count-1,batch_sample.tile_loc[::-1])
                            mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                            # mask_size_up , 16 to 256
                            k, l = mask.shape
                            for i in range(k):
                                for j in range(l):
                                    for o in range(16):
                                        for p in range(16):
                                            mask_size_up[i*16+o,j*16+p] = mask[i][j]

                    else:
                        x = 0
                        y = 0
                        
                        img = tiles.get_tile(tiles.level_count-1,(x,y))


                if img.size != (patch_size,patch_size):
                    print('this tisuue shape is not ',(patch_size,patch_size))
                    img = Image.new('RGB', (patch_size,patch_size))
                    mask_size_up = np.zeros((patch_size,patch_size))

                images.append(np.array(img))
                masks.append(mask_size_up)
            X_train = np.array(images)
            y_train = np.array(masks)
            y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], patch_size, patch_size, 2) 

            X_train, y_train = next(ImageDataGenerator(
                rotation_range=45,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range =(0.3,1.)).flow(X_train,y=y_train,batch_size=batch_size))
            yield X_train, y_train


                    



import gc
import psutil
from model import find_patches_from_slide, predict_from_model, simple_model

model = simple_model(pretrained_weights='s_1.h5')

# # Data Path Load
def read_data_path():
    image_paths = []
    with open('train.txt','r') as f:
        for line in f:
            line = line.rstrip('\n')
            image_paths.append(line)
    #print('image_path # : ',len(image_paths))

    tumor_mask_paths = []

    with open('train_mask.txt','r') as f:
        for line in f:
            line = line.rstrip('\n')
            tumor_mask_paths.append(line)
    #print('mask_patch # : ',len(tumor_mask_paths))
    
    return image_paths, tumor_mask_paths
def read_test_data_path():
    image_paths = []
    with open('test.txt','r') as f:
        for line in f:
            line = line.rstrip('\n')
            image_paths.append(line)
    #print('image_path # : ',len(image_paths))
    
    return image_paths

image_paths, tumor_mask_paths = read_data_path()


columns = ['is_tissue','slide_path','is_tumor','is_all_tumor','tile_loc']
# 5, 10, 15정도

slide_list_1 = [29,44,30,21,36,1,7,23,24,27,15,41,32,37,10,39,5,50,53,47,45,18,48,46,22,40,49,14,11,52,3,20,17,16,8,13,0,2,25,9,38,34,31,42,12,28,6,43,19,26,51]
slide_list_2 = [102,104,144,55,125,56,54,65,139,82,73,108,107,117,106,103,105,151,75,100,156,113,150,88,84,122,93,118,87,78,116,98,72,131,109,58,101,69,94,74,64,140,92,154,99,60,86,146,68,112,71,136,59,91,143,132,124,85]
slide_list_3 = [95,120,81,77,97,96,110,83,152,128,149,155,153,111,57,138,134,135,114,76,123,90,121,61,147,148,119,142,66,137,63,80,70,79,115,133,129,141]
# ALL SAMPLE DATA



data_start_time = datetime.now()

all_samples = pd.DataFrame(columns = columns)
# negative samples add
for i in range(len(slide_list_1)):
    image_path = image_paths[slide_list_1[i]]
    mask_path = tumor_mask_paths[slide_list_1[i]]
    samples = find_patches_from_slide(image_path,mask_path)
    samples = samples.sample(2000,random_state = 42,replace=True)
    all_samples = all_samples.append(samples)

for i in range(len(slide_list_2)):
    image_path = image_paths[slide_list_2[i]]
    mask_path = tumor_mask_paths[slide_list_2[i]]
    samples = find_patches_from_slide(image_path,mask_path)
    tumor_samples = samples[samples.is_tumor == True]
    tumor_samples = tumor_samples.sample(1000, random_state=42,replace=True)
    non_tumor_samples = samples[samples.is_tumor == False]
    non_tumor_samples = non_tumor_samples.sample(1000, random_state=42,replace=True)
    samples = tumor_samples.append(non_tumor_samples)
    all_samples = all_samples.append(samples)

for i in range(len(slide_list_3)):
    image_path = image_paths[slide_list_3[i]]
    mask_path = tumor_mask_paths[slide_list_3[i]]
    samples = find_patches_from_slide(image_path,mask_path)    
    tumor_samples = samples[samples.is_tumor == True]
    non_tumor_samples = samples[samples.is_tumor == False]
    non_tumor_samples = non_tumor_samples.sample(1000, random_state=42,replace=True)
    samples = tumor_samples.append(non_tumor_samples)
    all_samples = all_samples.append(samples)


print("-----------samples completed------------")
print("all_samples # : ", len(all_samples))
all_samples = all_samples.sample(frac=1)
all_samples = all_samples.sample(50000, random_state=42)
all_samples.reset_index(drop=True, inplace=True)
data_end_time = datetime.now()
print("samples completed time: %.1f minutes" % ((data_end_time - data_start_time).seconds / 60,))


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(all_samples, all_samples["is_tumor"]):
        train_samples = all_samples.loc[train_index]
        validation_samples = all_samples.loc[test_index]

train_generator = gen_imgs(train_samples, BATCH_SIZE)
validation_generator = gen_imgs(validation_samples, BATCH_SIZE)

train_start_time = datetime.now()
model.fit_generator(train_generator, np.ceil(len(train_samples) / BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=np.ceil(len(validation_samples) / BATCH_SIZE),
    epochs=N_EPOCHS)

train_end_time = datetime.now()
print("Model training time: %.1f minutes" % ((train_end_time - train_start_time).seconds / 60,))


path = '/data/model'
model.save(path+'/u_1.h5')
print('model save completed')