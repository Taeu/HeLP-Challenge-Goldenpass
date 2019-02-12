import os
import csv
import cv2
import openslide
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from openslide.deepzoom import DeepZoomGenerator
from common import find_patches_from_slide
import warnings

def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


file_handles = []
def generator(samples,
              slide_paths,
              truth_paths,
              batch_size,
              patch_size=256,
              shuffle=True):
    '''The generator for DataSet
        Args:
            - samples: DataFrame of samples
            - slide_paths: paths of all slides 
            - truth_paths: paths of all truth(masks)
            - batch_size: mini-batch size
            - patch_size: patch size for samples
            - shuffle: bool, if True shuffle samples
        Returns(yield):
            - train_x: train dataset → [batch_size, patch_size, patch_size, 3]
            - train_y: train labelset → [batch_size, patch_size, patch_size, 2]'''
    
    # 4개씩 묶은 slide path
    slide0 = openslide.open_slide(slide_paths[0])
    slide1 = openslide.open_slide(slide_paths[1])
    slide2 = openslide.open_slide(slide_paths[2])
    slide3 = openslide.open_slide(slide_paths[3])
    file_handles.append(slide0)
    file_handles.append(slide1)
    file_handles.append(slide2)
    file_handles.append(slide3)

    # tiles
    tiles0 = DeepZoomGenerator(slide0, tile_size=patch_size, overlap=0, limit_bounds=False)
    tiles1 = DeepZoomGenerator(slide1, tile_size=patch_size, overlap=0, limit_bounds=False)
    tiles2 = DeepZoomGenerator(slide2, tile_size=patch_size, overlap=0, limit_bounds=False)
    tiles3 = DeepZoomGenerator(slide3, tile_size=patch_size, overlap=0, limit_bounds=False)

    start_x0, start_y0 = 0, 0
    start_x1, start_y1 = 0, 0
    start_x2, start_y2 = 0, 0
    start_x3, start_y3 = 0, 0
    if 'pos' in slide_paths[0]:
        start_x0 = int(slide0.properties.get('openslide.bounds-x', 0)) / patch_size
        start_y0 = int(slide0.properties.get('openslide.bounds-y', 0)) / patch_size
        truth0 = openslide.open_slide(truth_paths[0])
        truth_tiles0 = DeepZoomGenerator(truth0, tile_size=16,overlap=0, limit_bounds=False)
    
    if 'pos' in slide_paths[1]: 
        start_x1 = int(slide1.properties.get('openslide.bounds-x', 0)) / patch_size
        start_y1 = int(slide1.properties.get('openslide.bounds-y', 0)) / patch_size
        truth1 = openslide.open_slide(truth_paths[1])
        truth_tiles1 = DeepZoomGenerator(truth1, tile_size=16,overlap=0, limit_bounds=False)
        
    if 'pos' in slide_paths[2]:
        start_x2 = int(slide2.properties.get('openslide.bounds-x', 0)) / patch_size
        start_y2 = int(slide2.properties.get('openslide.bounds-y', 0)) / patch_size
        truth2 = openslide.open_slide(truth_paths[2])
        truth_tiles2 = DeepZoomGenerator(truth2, tile_size=16, overlap=0, limit_bounds=False)
        
    if 'pos' in slide_paths[3]:
        start_x3 = int(slide3.properties.get('openslide.bounds-x', 0)) / patch_size
        start_y3 = int(slide3.properties.get('openslide.bounds-y', 0)) / patch_size
        truth3 = openslide.open_slide(truth_paths[3])
        truth_tiles3 = DeepZoomGenerator(truth3, tile_size=16, overlap=0, limit_bounds=False)
        
    num_samples = len(samples)
    while 1:
        if shuffle:
            samples = samples.sample(frac=1)  # shuffling

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]

            batch_tiles, batch_masks = [], []
            for slide_path, (y, x) in zip(batch_samples['slide_path'].values, 
                                          batch_samples['tile_loc'].values):
                
                mask_tile_zoom = np.zeros((patch_size,patch_size))
                if slide_path == slide_paths[0]:
                    img = tiles0.get_tile(tiles0.level_count-1, (x+start_x0, y+start_y0))
                    if 'pos' in slide_path:
                        mask_tile = truth_tiles0.get_tile(truth_tiles0.level_count-1, (x, y))
                        mask_tile = (cv2.cvtColor(np.array(mask_tile), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        # mask_size_up , 16 to 256
                        k, l = mask_tile.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_tile_zoom[i*16+o,j*16+p] = mask_tile[i][j]
                        
                elif slide_path == slide_paths[1]:
                    img = tiles1.get_tile(tiles1.level_count-1, (x+start_x1, y+start_y1))
                    if 'pos' in slide_path:
                        mask_tile = truth_tiles1.get_tile(truth_tiles1.level_count-1, (x, y))
                        mask_tile = (cv2.cvtColor(np.array(mask_tile), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        # mask_size_up , 16 to 256
                        k, l = mask_tile.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_tile_zoom[i*16+o,j*16+p] = mask_tile[i][j]
                
                elif slide_path == slide_paths[2]:
                    img = tiles2.get_tile(tiles2.level_count-1, (x+start_x2, y+start_y2))
                    if 'pos' in slide_path:
                        mask_tile = truth_tiles2.get_tile(truth_tiles2.level_count-1, (x, y))
                        mask_tile = (cv2.cvtColor(np.array(mask_tile), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        # mask_size_up , 16 to 256
                        k, l = mask_tile.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_tile_zoom[i*16+o,j*16+p] = mask_tile[i][j]

                elif slide_path == slide_paths[3]:
                    img = tiles3.get_tile(tiles3.level_count-1, (x+start_x3, y+start_y3))
                    if 'pos' in slide_path:
                        mask_tile = truth_tiles3.get_tile(truth_tiles3.level_count-1, (x, y))
                        mask_tile = (cv2.cvtColor(np.array(mask_tile), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        # mask_size_up , 16 to 256
                        k, l = mask_tile.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_tile_zoom[i*16+o,j*16+p] = mask_tile[i][j]

                
                if img.size != (patch_size, patch_size):
                    img = Image.new('RGB', (patch_size, patch_size))
                    mask_tile_zoom = np.zeros((patch_size, patch_size))
                    
                batch_tiles.append(np.array(img))
                batch_masks.append(mask_tile_zoom)
                
            # train_x & train_y
            train_x = np.array(batch_tiles)
            train_y = to_categorical(np.array(batch_masks), num_classes=2)
            
            yield train_x, train_y


def create_model(patch_size=256, pre_trained_path=False):
    # Build U-Net model
    inputs = layers.Input(shape=(patch_size, patch_size, 3), dtype='float32', name='inputs')
    inputs_norm = layers.Lambda(lambda x: x/255. - .5)(inputs)

    # Conv layers
    conv1 = layers.Conv2D(16, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(inputs_norm)
    conv1 = layers.Dropout(0.1)(conv1)
    conv1 = layers.Conv2D(16, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D()(conv1)

    conv2 = layers.Conv2D(32, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Dropout(0.1)(conv2)
    conv2 = layers.Conv2D(32, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D()(conv2)

    conv3 = layers.Conv2D(64, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Dropout(0.2)(conv3)
    conv3 = layers.Conv2D(64, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D()(conv3)

    conv4 = layers.Conv2D(128, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Dropout(0.2)(conv4)
    conv4 = layers.Conv2D(128, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv4)
    pool4 = layers.MaxPooling2D()(conv4)

    conv5 = layers.Conv2D(256, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Dropout(0.3)(conv5)
    conv5 = layers.Conv2D(256, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv5)

    # Up-Conv layers
    up_conv6 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(conv5)
    up_conv6 = layers.concatenate([up_conv6, conv4])
    conv6 = layers.Conv2D(128, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(up_conv6)
    conv6 = layers.Dropout(0.2)(conv6)
    conv6 = layers.Conv2D(128, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv6)

    up_conv7 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv6)
    up_conv7 = layers.concatenate([up_conv7, conv3])
    conv7 = layers.Conv2D(64, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(up_conv7)
    conv7 = layers.Dropout(0.2)(conv7)
    conv7 = layers.Conv2D(64, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv7)

    up_conv8 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(conv7)
    up_conv8 = layers.concatenate([up_conv8, conv2])
    conv8 = layers.Conv2D(32, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(up_conv8)
    conv8 = layers.Dropout(0.1)(conv8)
    conv8 = layers.Conv2D(32, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv8)

    up_conv9 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(conv8)
    up_conv9 = layers.concatenate([up_conv9, conv1])
    conv9 = layers.Conv2D(16, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(up_conv9)
    conv9 = layers.Dropout(0.1)(conv9)
    conv9 = layers.Conv2D(16, 3, padding='same', 
                          activation='relu', kernel_initializer='he_normal')(conv9)

    outputs = layers.Conv2D(2, 1, activation='softmax', 
                           kernel_initializer='he_normal')(conv9)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adamax(),
                  loss='categorical_crossentropy', 
                  metrics=['acc'])
    
    if pre_trained_path:
        model = models.load_model(pre_trained_path)
    
    return model


model = create_model(pre_trained_path='./unet.h5')


def get_data_path():
    slide_paths, mask_paths = {}, {}
    with open('./train.txt', 'r') as f:
        for idx, line in enumerate(f):
            path = line.rstrip('\n')
            slide_paths[idx] = path
    
    with open('./train_mask.txt', 'r') as f:
        for idx, line in enumerate(f):
            path = line.rstrip('\n')
            mask_paths[idx] = path
            
    return slide_paths, mask_paths


slide_paths, mask_paths = get_data_path()

slide_4_list_1 = [[102,104,29,44],[144,55,30,18],[54,65,21,36],[139,82,1,49],[105,151,15,2],[75,100,41,9],[156,113,32,37]]
slide_4_list_2 = [[109,58,14,28],[101,69,11,43],[94,74,3,20],[64,140,17,16],[92,154,8,26],[99,60,0,33],[86,146,25,19],[68,112,38,51],
                 [71,136,31,4],[59,91,12,6]]
slide_4_list_3 = [[143,132,124,85],[95,120,81,77],[97,96,110,83],[152,128,149,155],[153,111,57,138],[134,135,114,76],
                  [123,90,121,61],[147,148,119,142],[66,137,63,80],[70,79,115,133],[129,141,127,145]]
slide_4_test = [[55,55, 0, 0]]

columns = ['is_tissue','slide_path','is_tumor','is_all_tumor','tile_loc']

batch_size = 32
n_epochs = 30
print('======== Start Train ========')
for slides in slide_4_list_1:
    sample_group_df = pd.DataFrame(
            columns=['is_tissue','slide_path','is_tumor','is_all_tumor','tile_loc'])
    
    group_slide_path, group_mask_path = [], []
    for idx in slides:
        slide_path, truth_path = slide_paths[idx], mask_paths[idx]
        samples = find_patches_from_slide(slide_path, truth_path)
        sample_group_df = sample_group_df.append(samples)
        group_slide_path.append(slide_path)
        group_mask_path.append(truth_path)
        
    num_samples = len(sample_group_df)
    if num_samples > 10000:
        num_samples = 10000
    
    samples = sample_group_df.sample(num_samples, random_state=42)
    samples.reset_index(drop=True, inplace=True)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(samples, samples["is_tumor"]):
            train_samples = samples.loc[train_index]
            validation_samples = samples.loc[test_index]
            
    train_gen = generator(train_samples, group_slide_path, group_mask_path, batch_size)
    val_gen = generator(validation_samples, group_slide_path, group_mask_path, batch_size)
    
    model.fit_generator(train_gen, 
                        steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=np.ceil(len(validation_samples)/batch_size),
                        verbose=2)
    
    for fh in file_handles:
        fh.close()
    file_handles = []

model.save('/data/model/unet.h5')