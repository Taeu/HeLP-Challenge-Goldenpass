import os
import gc
import csv
import time
import cv2
import openslide
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from keras import layers, models
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from openslide.deepzoom import DeepZoomGenerator
from model import create_model
from common import find_patches_from_slide

print('**************************** run train.py ***************************************')

file_handles = []
def generator(samples,
              slide_paths,
              truth_paths,
              batch_size,
              patch_size=256,
              shuffle=True):
    
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


# # # model = create_model()
print('============ Create model for Training ==============')
model = create_model()
# model = create_model(pretrained_weights='./model/unet-v2.h5')
print('============ Model Loaded for Training ==============')


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

slide_4_list_1 = [
    [102,104,29,44], [144,55,30,18],
    [54,65,21,36], [139,82,1,49],
    [73,108,7,23], [106,103,27,13],
    [125,56,40,40], [105,151,15,2],
    [75,100,41,9], [156,113,32,37],
    [150,88,39,10], [84,122,5,50],
    [93,118,53,47], [107,117,24,52],
    [87,78,45,34], [116,98,48,46],[72,131,22,42]
]

# slide_4_test = [[55,55,0,0]]
# slide_4_docker_test = [[102,104,29,44]]

columns = ['is_tissue','slide_path','is_tumor','is_all_tumor','tile_loc']

batch_size = 32
n_epochs = 1
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
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
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

    gc.collect()
    del train_gen
    del val_gen


print('========= Model saving... =======')
model.save_weights('/data/model/unet.h5')
print('========= Model saved!!!! ========')
print('********************** Train finished **********************')