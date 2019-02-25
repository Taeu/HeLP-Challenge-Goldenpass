import os
import csv
import time
import cv2
import openslide
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from scipy import stats
from keras import layers, models
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from openslide.deepzoom import DeepZoomGenerator
from model import create_model
from preprocessing import open_slide, create_tiles
from preprocessing import create_patches


print('********************** run inference.py *********************')

print('============ Pretrained model Load... =============')
model = create_model(pretrained_weights='/data/model/incept-unet.h5')
# model = create_model(pretrained_weights='/data/model/incept-unet.h5')
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


test_paths = get_test_path()

patch_size = 256
batch_size = 32

print('======== Start Inference ========')
slide_ids, slide_preds = [], []
for slide_id, slide_path in test_paths.items():
    samples = create_patches(slide_path, 'test')
    
    num_samples = len(samples)
    if num_samples > 5000:
        num_samples = 3000
    
    samples = samples.sample(num_samples, random_state=42)
    samples.reset_index(drop=True, inplace=True)
    
    test_gen = test_generator(samples, slide_path, batch_size)
    test_steps = np.ceil(len(samples) / batch_size)
    
    predicts = model.predict_generator(test_gen,
                                       steps=test_steps)
    preds = predicts[:, :, :, 1]
    pred = preds.reshape((num_samples, -1))
    preds = stats.mode(preds)
    preds = preds[0]
    pred = np.average(preds)

    slide_ids.append(slide_id)
    slide_preds.append(pred)
    
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