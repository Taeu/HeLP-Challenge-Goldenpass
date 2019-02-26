# coding: utf-8
import os.path as osp
import openslide
from pathlib import Path
import numpy as np
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')

from skimage.filters import threshold_otsu
from openslide.deepzoom import DeepZoomGenerator
import cv2
from keras.utils.np_utils import to_categorical
from keras.models import load_model

# Unet
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# train
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

import math
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import keras

# save tiles
import gc
import cv2
import psutil
from model import find_patches_from_slide, predict_batch_from_model, predict_from_model, InceptionV3

PATCH_SIZE = 256
IS_TRAIN = True
BATCH_SIZE = 32
N_EPOCHS = 2

NUM_CLASSES = 2 # not_tumor, tumor
file_handles=[]

print('=============== make directory for tiles =================')
os.mkdir('/data/model/patches')
os.mkdir('/data/model/patches/train')
os.mkdir('/data/model/patches/train/neg')
os.mkdir('/data/model/patches/train/pos')
os.mkdir('/data/model/patches/val')
os.mkdir('/data/model/patches/val/neg')
os.mkdir('/data/model/patches/val/pos')
print('============ directory created!! =========================')



def save_train_imgs(samples,patch_size = PATCH_SIZE):
    num_samples = len(samples)
    for idx in range(num_samples):
        for _,batch_sample in samples.iloc[idx:idx+1].iterrows():
            slide_path = batch_sample.slide_path
        slide_name = 's'
        slide_contains_tumor = 'pos' in slide_path
        with openslide.open_slide(slide_path) as slide:
            tiles = DeepZoomGenerator(slide, tile_size = patch_size, overlap=0, limit_bounds = False)
            x,y = batch_sample.tile_loc[::-1]
            if slide_contains_tumor:
                start_x = int(slide.properties.get('openslide.bounds-x',0))
                start_y = int(slide.properties.get('openslide.bounds-y',0))
                start_x = start_x / patch_size
                start_y = start_y / patch_size
                x += start_x
                y += start_y

                img = tiles.get_tile(tiles.level_count-1,(x,y))
                img.save('/data/model/patches/train/pos/{}_pos_{}.png'.format(slide_name, idx))

            else:
                img = tiles.get_tile(tiles.level_count-1,(x,y))
                img.save('/data/model/patches/train/neg/{}_neg_{}.png'.format(slide_name, idx))

    print('num_samples_train : ',num_samples)
    print('train img save completed')
def save_valid_imgs(samples,patch_size = PATCH_SIZE):
    num_samples = len(samples)
    for idx in range(num_samples):
        for _,batch_sample in samples.iloc[idx:idx+1].iterrows():
            slide_path = batch_sample.slide_path
        print(slide_path)
        slide_name = 's'
        slide_contains_tumor = 'pos' in slide_path
        with openslide.open_slide(slide_path) as slide:
            tiles = DeepZoomGenerator(slide, tile_size = patch_size, overlap=0, limit_bounds = False)
            x,y = batch_sample.tile_loc[::-1]
            if slide_contains_tumor:
                start_x = int(slide.properties.get('openslide.bounds-x',0))
                start_y = int(slide.properties.get('openslide.bounds-y',0))
                start_x = start_x / patch_size
                start_y = start_y / patch_size
                x += start_x
                y += start_y

                img = tiles.get_tile(tiles.level_count-1,(x,y))
                img.save('/data/model/patches/val/pos/{}_pos_{}.png'.format(slide_name, idx))

            else:
                img = tiles.get_tile(tiles.level_count-1,(x,y))
                img.save('/data/model/patches/val/neg/{}_neg_{}.png'.format(slide_name, idx))

    print('num_samples_val : ',num_samples)
    print('val img save completed')
def generator(gen):
    while 1:
        train_x, labels = next(gen)
        train_y = []
        for label in labels:
            if label == 0:
                train_y.append(np.zeros((256, 256)))
            else:
                train_y.append(np.ones((256, 256), dtype='float'))
        train_y = np.array(train_y)
        train_y = to_categorical(train_y, num_classes=2)
        yield train_x, train_y
        
########### should add pretrained model

model = InceptionV3(include_top=True,
                weights='i_5.h5',
                input_tensor=None,
                input_shape=(256,256,3),
                pooling=None,
                classes=2)

#model = simple_model(pretrained_weights='s_1.h5')
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
samples = all_samples.sample(5000, random_state=42)
samples.reset_index(drop=True, inplace=True)
data_end_time = datetime.now()
print("samples completed time: %.1f minutes" % ((data_end_time - data_start_time).seconds / 60,))

callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor = 'loss',
        factor = 0.5,
        patience = 3,
        min_lr = 0.00001,
        verbose = 1,
    )]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(samples, samples["is_tumor"]):
    train_samples = samples.loc[train_index]
    validation_samples = samples.loc[test_index]

data_start_time = datetime.now()
print("-----------samples img creating------------")
save_train_imgs(train_samples)
save_valid_imgs(validation_samples)
print("-----------samples img created-------------")
data_end_time = datetime.now()
print("samples img completed time: %.1f minutes" % ((data_end_time - data_start_time).seconds / 60,))

train_datagen = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range =(0.5, 1.),)
test_datagen = ImageDataGenerator()

train_dir = '/data/model/patches/train/'
val_dir = '/data/model/patches/val/'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256), 
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

train_gen = generator(train_generator)
val_gen = generator(validation_generator)

train_start_time = datetime.now()
model.fit_generator(train_gen, np.ceil(len(train_samples) / BATCH_SIZE),
    validation_data=val_gen,
    validation_steps=10,
    epochs=N_EPOCHS,
    callbacks = callbacks_list,
    verbose=2,
    )

train_end_time = datetime.now()
print("Model training time: %.1f minutes" % ((train_end_time - train_start_time).seconds / 60,))


path = '/data/model'
model.save(path+'/i_1.h5')
print('model save completed')

