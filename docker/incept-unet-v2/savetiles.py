import os
import gc
import cv2
import openslide
import numpy as np
import pandas as pd

from PIL import Image
from openslide.deepzoom import DeepZoomGenerator
from preprocessing import get_train_path, get_pos_path
from preprocessing import open_slide, create_tiles
from preprocessing import create_patches


print('**************** savetiles.py start!! ******************')

slide_paths, mask_paths = get_pos_path()

def save_patches(samples,
                 slide_path,
                 truth_path,
                 patch_size=256,
                 is_training=True):
    
    slide_name = os.path.split(slide_path)[1].split('.')[0]

    slide = open_slide(slide_path)
    tiles = create_tiles(slide, tile_size=patch_size)

    start_x, start_y = 0, 0
    if 'pos' in slide_path:
        start_x = int(slide.properties.get('openslide.bounds-x', 0)) / patch_size
        start_y = int(slide.properties.get('openslide.bounds-y', 0)) / patch_size
        truth = open_slide(truth_path)
        truth_tiles = create_tiles(truth, tile_size=16)
        
#     samples = samples.sample(num_samples, random_state=42)  # shuffling
    num_tumor_dict = samples['is_tumor'].value_counts().to_dict()
    num_false = num_tumor_dict[True]
    num_true = num_tumor_dict[False]

    if num_true > 300:
        num_false, num_true = 300, 300
    elif num_true < 300:
        num_false = 300

    tumor_samples = samples.query('is_tumor == True').sample(num_true, random_state=42)
    false_samples = samples.query('is_tumor == False').sample(num_false, random_state=42)
    train_samples = tumor_samples.append(false_samples)
    
    tumor_samples = samples.query('is_tumor == True').sample(45)
    false_samples = samples.query('is_tumor == False').sample(45)
    val_samples = tumor_samples.append(false_samples)
    
    for idx, (y, x) in enumerate(train_samples['tile_loc'].values):
        img = tiles.get_tile(tiles.level_count-1, (x+start_x, y+start_y))
        if 'pos' in slide_path:
            mask_tile = truth_tiles.get_tile(truth_tiles.level_count-1, (x, y))
            mask_tile = (cv2.cvtColor(np.array(mask_tile), cv2.COLOR_RGB2GRAY) > 0).astype(int)
            if mask_tile[0][0] == 0:
                img.save('./tiles/train/neg/{}_neg_{}.png'.format(slide_name, idx))
            else:
                img.save('./tiles/train/pos/{}_pos_{}.png'.format(slide_name, idx))
                
    for idx, (y, x) in enumerate(val_samples['tile_loc'].values):
        img = tiles.get_tile(tiles.level_count-1, (x+start_x, y+start_y))
        if 'pos' in slide_path:
            mask_tile = truth_tiles.get_tile(truth_tiles.level_count-1, (x, y))
            mask_tile = (cv2.cvtColor(np.array(mask_tile), cv2.COLOR_RGB2GRAY) > 0).astype(int)
            if mask_tile[0][0] == 0:
                img.save('./tiles/val/neg/{}_neg_{}.png'.format(slide_name, idx))
            else:
                img.save('./tiles/val/pos/{}_pos_{}.png'.format(slide_name, idx))


for slide_path, mask_path, in zip(slide_paths.values(), mask_paths.values()):
    samples = create_patches(slide_path, mask_path)
    save_patches(samples, slide_path, mask_path)
    print('{} finished!!'.format(os.path.split(slide_path)))


print('****************** savetiles.py finished! ***********************')