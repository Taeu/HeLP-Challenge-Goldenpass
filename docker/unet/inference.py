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
    with open('./test.txt', 'r') as f:
        for line in f:
            path = line.rstrip('\n')
            _, fname = os.path.split(path)
            fname = fname.split('.')[0]
            slide_paths[fname] = path
                
    return slide_paths

test_paths = get_test_path()

patch_size = 256
batch_size = 32

# test_paths = {'Slide003': '../../data/test/Slide003.mrxs'}
print('======== Load model ========')
model = models.load_model('/data/model/unet.h5')

print('======== Start Inference ========')
slide_ids, slide_preds = [], []
for slide_id, slide_path in test_paths.items():
    samples = find_patches_from_slide(slide_path, 'test')
    
    num_samples = len(samples)
    
    samples = samples.sample(1000, random_state=42)
    samples.reset_index(drop=True, inplace=True)
    
    test_gen = test_generator(samples, slide_path, batch_size)
    test_steps = np.ceil(len(samples) / batch_size)
    
    predicts = model.predict_generator(test_gen,
                                       steps=test_steps)
    predict = np.max(predicts[:, :, :, 1])
    slide_ids.append(slide_id)
    slide_preds.append(predict)
    
    for fh in file_handles:
        fh.close()
    file_handles = []


# Create output.csv
output = pd.DataFrame()
output['slide_id'] = slide_ids
output['slide_pred'] = slide_preds
output.to_csv('/data/output/output.csv', index=False, header=False)