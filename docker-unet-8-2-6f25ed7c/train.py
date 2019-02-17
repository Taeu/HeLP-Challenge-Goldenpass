
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
from keras import optimizers

import gc
import psutil

PATCH_SIZE = 256
IS_TRAIN = True

from model import find_patches_from_slide, gen_imgs, unet, predict_batch_from_model, predict_from_model

NUM_CLASSES = 2 # not_tumor, tumor
file_handles=[]

from PIL import ImageEnhance as ie

model = unet(pretrained_weights='u_3.h5')


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

slide_4_list_1 = [[102,104,29,44],[144,55,30,18],[54,65,21,36],[139,82,1,49],[73,108,7,23],[106,103,27,13],[125,56,40,40],
               [105,151,15,2],[75,100,41,9],[156,113,32,37],[150,88,39,10],[84,122,5,50],[93,118,53,47],[107,117,24,52],[87,78,45,34],[116,98,48,46],
            [72,131,22,42]]
slide_4_list_2 = [[101,69,11,43],[94,74,3,20],[64,140,17,16],[71,136,31,4],[59,91,12,6],[92,154,8,26],[99,60,0,33],[86,146,25,19],[68,112,38,51],
                 [109,58,14,28]]
slide_4_list_3 = [[143,132,124,85],[95,120,81,77],[97,96,110,83],[152,128,149,155],[153,111,57,138],[134,135,114,76],
                  [123,90,121,61],[147,148,119,142],[66,137,63,80],[70,79,115,133],[129,141,127,145]]
slide_4_test = [[55,55,0,0]]
columns = ['is_tissue','slide_path','is_tumor','is_all_tumor','tile_loc']


BATCH_SIZE = 128
N_EPOCHS = 10


for i in range(len(slide_4_list_1)):
    # [1] dataset , 2 pos, 2 neg, mean ratio = 3:1
    print(i,'th training\n')
    four_samples = pd.DataFrame(columns = columns)
    four_image_path = list()
    four_mask_path = list()    
    for j in range(4):
        image_path = image_paths[slide_4_list_1[i][j]] # 이 부분은 data 읽을때 고치자 ( [1:] 빼야함)
        mask_path = tumor_mask_paths[slide_4_list_1[i][j]] # 이 부분은 data 읽을때 고치자
        samples = find_patches_from_slide(image_path, mask_path)
        
        four_samples = four_samples.append(samples)   
        four_image_path.append(image_path)
        four_mask_path.append(mask_path)
        
    # train sample size
    NUM_SAMPLES = len(four_samples)
    if NUM_SAMPLES > 10000:
        NUM_SAMPLES = 10000
    
    samples = four_samples.sample(NUM_SAMPLES, random_state=42)
    samples.reset_index(drop=True, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(samples, samples["is_tumor"]):
            train_samples = samples.loc[train_index]
            validation_samples = samples.loc[test_index]

    train_generator = gen_imgs(four_image_path,four_mask_path,train_samples, BATCH_SIZE)
    validation_generator = gen_imgs(four_image_path,four_mask_path,validation_samples, BATCH_SIZE)
    
    train_start_time = datetime.now()
    history = model.fit_generator(train_generator, np.ceil(len(train_samples) / BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=np.ceil(len(validation_samples) / BATCH_SIZE),
        epochs=N_EPOCHS, verbose=2)

    train_end_time = datetime.now()
    print("Model training time: %.1f minutes" % ((train_end_time - train_start_time).seconds / 60,))
    # split
    # data gen : all_image_path, all_mask_path
    for fh in file_handles:
        fh.close()
    file_handles = []

    gc.collect()
    del train_generator
    del validation_generator
    



path = '/data/model'
model.save(path+'/unet555.h5')
print('model save completed')

