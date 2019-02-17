import os
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


print('********************** run inference.py *********************')

print('============ Pretrained model Load... =============')
model = create_model(pretrained_weights='/data/model/unet.h5')
print('============ Model Loaded ==========')

file_handles = []
def test_generator(samples,
                   slide_path,
                   batch_size,
                   patch_size=256,
                   shuffle=True):
    
    slide = openslide.open_slide(slide_path)
    file_handles.append(slide)

    # tiles
    tiles = DeepZoomGenerator(slide, tile_size=patch_size, overlap=0, limit_bounds=False)

    num_samples = len(samples)
    while 1:
        if shuffle:
            samples = samples.sample(frac=1)  # shuffling

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]

            batch_tiles= []
            for slide_path, (y, x) in zip(batch_samples['slide_path'].values, 
                                          batch_samples['tile_loc'].values):
                img = tiles.get_tile(tiles.level_count-1, (x, y))             
                
                if img.size != (patch_size, patch_size):
                    img = Image.new('RGB', (patch_size, patch_size))
                    
                batch_tiles.append(np.array(img))
                
            # train_x
            test_x = np.array(batch_tiles)
            yield test_x

def get_test_path():
    slide_paths = {}
    files = os.listdir('/data/test/')
    for file_ in files:
        if 'mrxs' in file_:
            fname = file_.split('.')[0]
            slide_paths[fname] = '/data/test/' + file_
        
    return slide_paths
# def get_test_path():
    # slide_paths = {}
    # with open('./test2.txt', 'r') as f:
    #     for line in f:
    #         path = line.rstrip('\n')
    #         _, fname = os.path.split(path)
    #         fname = fname.split('.')[0]
    #         slide_paths[fname] = path
                
    # return slide_paths

test_paths = get_test_path()

patch_size = 256
batch_size = 32


print('======== Start Inference ========')
slide_ids, slide_preds = [], []
for slide_id, slide_path in test_paths.items():
    samples = find_patches_from_slide(slide_path, 'test')
    
    num_samples = len(samples)
    if num_samples > 5000:
        num_samples = 1
    
    samples = samples.sample(num_samples, random_state=42)
    samples.reset_index(drop=True, inplace=True)
    
    test_gen = test_generator(samples, slide_path, batch_size)
    test_steps = np.ceil(len(samples) / batch_size)
    
    predicts = model.predict_generator(test_gen,
                                       steps=test_steps)
    predict = np.max(predicts[:, :, :, 1])
    # print('{},{}'.format(slide_id, predict))
    slide_ids.append(slide_id)
    slide_preds.append(predict)
    
    for fh in file_handles:
        fh.close()
    file_handles = []

print('======== Finish Inference ========')

print('******************** Create output.csv ************************')
# Create output.csv
output = pd.DataFrame()
output['slide_id'] = slide_ids
output['slide_pred'] = slide_preds
output = output.sort_values('slide_id')
output.to_csv('/data/output/output.csv', index=False, header=False)

slide_ids = output['slide_id'].values
slide_preds = output['slide_pred'].values
for slide_id, slide_pred in zip(slide_ids, slide_preds):
    print('{},{}'.format(slide_id, slide_pred))