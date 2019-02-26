
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
def find_patches_from_slide(slide_path, truth_path, patch_size=PATCH_SIZE,filter_non_tissue=True,filter_only_all_tumor=True):
    
    slide_contains_tumor = 'pos' in slide_path
    
    ############### read_region을 위한 start, level, size를 구함 #######################
    BOUNDS_OFFSET_PROPS = (openslide.PROPERTY_NAME_BOUNDS_X, openslide.PROPERTY_NAME_BOUNDS_Y)
    BOUNDS_SIZE_PROPS = (openslide.PROPERTY_NAME_BOUNDS_WIDTH, openslide.PROPERTY_NAME_BOUNDS_HEIGHT)


    if slide_contains_tumor:
        with openslide.open_slide(slide_path) as slide:
            start = (int(slide.properties.get('openslide.bounds-x',0)),int(slide.properties.get('openslide.bounds-y',0)))
            level = np.log2(patch_size) 
            level = int(level)
            
            size_scale = tuple(int(slide.properties.get(prop, l0_lim)) / l0_lim
                            for prop, l0_lim in zip(BOUNDS_SIZE_PROPS,
                            slide.dimensions))
            _l_dimensions = tuple(tuple(int(math.ceil(l_lim * scale))
                            for l_lim, scale in zip(l_size, size_scale))
                            for l_size in slide.level_dimensions)
            size = _l_dimensions[level]
            
            
            with openslide.open_slide(truth_path) as truth:
                print('truth dimensions: ',truth.dimensions)
                z_dimensions=[]
                z_size = truth.dimensions
                z_dimensions.append(z_size)
                while z_size[0] > 1 or z_size[1] > 1:
                    
                    z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
                    z_dimensions.append(z_size)
                print('truth_4_dimension_size:',z_dimensions[4]) # level-4
            size = z_dimensions[level-4]
            slide4 = slide.read_region(start,level,size)
            print('slide4_size',slide4.size)
    else :
        with openslide.open_slide(slide_path) as slide:
            start = (0,0)
            level = np.log2(patch_size) 
            level = int(level)
            
            size_scale = (1,1)
            _l_dimensions = tuple(tuple(int(math.ceil(l_lim * scale))
                            for l_lim, scale in zip(l_size, size_scale))
                            for l_size in slide.level_dimensions)
            size = _l_dimensions[level]
            
            slide4 = slide.read_region(start,level,size) 
    ####################################################################################
    
    
    # is_tissue 부분 
    slide4_grey = np.array(slide4.convert('L'))
    binary = slide4_grey > 0  # black이면 0임
    
    # 검은색 제외하고 흰색영역(배경이라고 여겨지는)에 대해서도 작업해주어야함.
    slide4_not_black = slide4_grey[slide4_grey>0]
    thresh = threshold_otsu(slide4_not_black)
    
    I, J = slide4_grey.shape
    for i in range(I):
        for j in range(J):
            if slide4_grey[i,j] > thresh :
                binary[i,j] = False
    patches = pd.DataFrame(pd.DataFrame(binary).stack())
    patches['is_tissue'] = patches[0]
    patches.drop(0, axis=1,inplace =True)
    patches.loc[:,'slide_path'] = slide_path
    

    if slide_contains_tumor:
        with openslide.open_slide(truth_path) as truth:
            thumbnail_truth = truth.get_thumbnail(size) 
        
        patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack())
        patches_y['is_tumor'] = patches_y[0] > 0
        
        # mask된 영역이 애매할 수도 있으므로
        patches_y['is_all_tumor'] = patches_y[0] == 255
        patches_y.drop(0, axis=1, inplace=True)
        samples = pd.concat([patches, patches_y], axis=1) #len(samples)
    else:
        samples = patches
        #dfmi.loc[:,('one','second')] = value
        samples.loc[:,'is_tumor'] = False
        samples.loc[:,'is_all_tumor'] = False
    
    if filter_non_tissue:
        samples = samples[samples.is_tissue == True] # remove patches with no tissue #samples = samples[samples.is_tissue == True]
    
    if filter_only_all_tumor :
        samples['tile_loc'] = list(samples.index)
        all_tissue_samples1 = samples[samples.is_tumor==False]
        all_tissue_samples1 = all_tissue_samples1.append(samples[samples.is_all_tumor==True])
        
        all_tissue_samples1.reset_index(inplace=True, drop=True)
    else :
        return samples
    
    return all_tissue_samples1

NUM_CLASSES = 2 # not_tumor, tumor
def gen_imgs(all_image_path, all_mask_path, samples, batch_size, patch_size = PATCH_SIZE, shuffle=True):
   
    num_samples = len(samples)
    # 특정 몇개의 slide만 open 해서 쓰기
    # 4개씩 묶었으니까 
  
    slide_path0 = all_image_path[0]
    slide_path1 = all_image_path[1]
    slide_path2 = all_image_path[2]
    slide_path3 = all_image_path[3]
    print('slide_path0 : ',slide_path0)
    print('slide_path1 : ',slide_path1)
    print('slide_path2 : ',slide_path2)
    print('slide_path3 : ',slide_path3)
    
    # slide 0~3 까지 미리 열어두기
    slide0 = openslide.open_slide(slide_path0)
    slide1 = openslide.open_slide(slide_path1)
    slide2 = openslide.open_slide(slide_path2)
    slide3 = openslide.open_slide(slide_path3)
    
    # with openslide.open_slide(slide_path) as slide
    tiles0 = DeepZoomGenerator(slide0,tile_size=patch_size, overlap=0, limit_bounds=False) 
    tiles1 = DeepZoomGenerator(slide1,tile_size=patch_size, overlap=0, limit_bounds=False)
    tiles2 = DeepZoomGenerator(slide2,tile_size=patch_size, overlap=0, limit_bounds=False)
    tiles3 = DeepZoomGenerator(slide3,tile_size=patch_size, overlap=0, limit_bounds=False)
    print(tiles0)
    print(tiles1)
    print(tiles2)
    print(tiles3)

    if 'pos' in slide_path0:
        start_x0 = int(slide0.properties.get('openslide.bounds-x',0))
        start_y0 = int(slide0.properties.get('openslide.bounds-y',0))
        start_x0 = start_x0 / patch_size
        start_y0 = start_y0 / patch_size
        
        truth0 = openslide.open_slide(all_mask_path[0])
        truth_tiles0 = DeepZoomGenerator(truth0, tile_size=16, overlap=0, limit_bounds=False) 
        
    else : 
        start_x0 = 0
        start_y0 = 0
    
    if 'pos' in slide_path1:
        start_x1 = int(slide1.properties.get('openslide.bounds-x',0))
        start_y1 = int(slide1.properties.get('openslide.bounds-y',0))
        start_x1 = start_x1 / patch_size
        start_y1 = start_y1 / patch_size
        
        truth1 = openslide.open_slide(all_mask_path[1])
        truth_tiles1 = DeepZoomGenerator(truth1, tile_size=16, overlap=0, limit_bounds=False) 
        
    else : 
        start_x1 = 0
        start_y1 = 0
    
    if 'pos' in slide_path2:
        start_x2 = int(slide2.properties.get('openslide.bounds-x',0))
        start_y2 = int(slide2.properties.get('openslide.bounds-y',0))
        start_x2 = start_x2 / patch_size
        start_y2 = start_y2 / patch_size
        
        truth2 = openslide.open_slide(all_mask_path[2])
        truth_tiles2 = DeepZoomGenerator(truth2, tile_size=16, overlap=0, limit_bounds=False) 
        
    else : 
        start_x2 = 0
        start_y2 = 0
        
    if 'pos' in slide_path3:
        start_x3 = int(slide3.properties.get('openslide.bounds-x',0))
        start_y3 = int(slide3.properties.get('openslide.bounds-y',0))
        start_x3 = start_x3 / patch_size
        start_y3 = start_y3 / patch_size
        
        truth3 = openslide.open_slide(all_mask_path[3])
        truth_tiles3 = DeepZoomGenerator(truth3, tile_size=16, overlap=0, limit_bounds=False) 
        
    else : 
        start_x3 = 0
        start_y3 = 0
    

    
    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1) # shuffle samples

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            masks = []
            for _, batch_sample in batch_samples.iterrows(): # 배치마다 deep zoom 하네 약간 비효율적
                
                # 여기서 하나씩 4개 체크해서 해당되는 부분으로 가야지. for 4번 돌리면서 가야한다.
                mask_size_up = np.zeros((patch_size,patch_size))
                a,b=mask_size_up.shape
                print(batch_sample.slide_path)
                if batch_sample.slide_path == slide_path0:
                    x, y = batch_sample.tile_loc[::-1]
                    x += start_x0
                    y += start_y0
                    img = tiles0.get_tile(tiles0.level_count-1, (x,y))
                    if 'pos' in slide_path0:
                        mask = truth_tiles0.get_tile(truth_tiles0.level_count-1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                            # mask_size_up , 16 to 256
                        k, l = mask.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_size_up[i*16+o,j*16+p] = mask[i][j]

                        
                        # for i in range(a):
                        #     for j in range(b) :
                        #         k = i//16
                        #         l = j//16
                        #         mask_size_up[i,j] = mask[k,l]
                    
                elif batch_sample.slide_path == slide_path1:
                    x, y = batch_sample.tile_loc[::-1]
                    x += start_x1
                    y += start_y1
                    img = tiles1.get_tile(tiles1.level_count-1, (x,y))
                    if 'pos' in slide_path1:
                        mask = truth_tiles1.get_tile(truth_tiles1.level_count-1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                            # mask_size_up , 16 to 256
                        k, l = mask.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_size_up[i*16+o,j*16+p] = mask[i][j]

                
                elif batch_sample.slide_path == slide_path2:
                    x, y = batch_sample.tile_loc[::-1]
                    x += start_x2
                    y += start_y2
                    img = tiles2.get_tile(tiles2.level_count-1, (x,y))
                    if 'pos' in slide_path2:
                        mask = truth_tiles2.get_tile(truth_tiles2.level_count-1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                            # mask_size_up , 16 to 256
                        k, l = mask.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_size_up[i*16+o,j*16+p] = mask[i][j]

                
                else:
                    x, y = batch_sample.tile_loc[::-1]
                    x += start_x3
                    y += start_y3
                    img = tiles3.get_tile(tiles3.level_count-1, (x,y))
                    if 'pos' in slide_path3:
                        mask = truth_tiles3.get_tile(truth_tiles3.level_count-1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                            # mask_size_up , 16 to 256
                        k, l = mask.shape
                        for i in range(k):
                            for j in range(l):
                                for o in range(16):
                                    for p in range(16):
                                        mask_size_up[i*16+o,j*16+p] = mask[i][j]

                
                # 여기에 조건문 넣자.
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
            
            #X_train, y_train = datagen().flow(X_train,y = y_train,batch_size = batch_size)
            X_train, y_train = next(ImageDataGenerator(
                rotation_range=90,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range =(0.25,1.)).flow(X_train,y=y_train,batch_size=batch_size))
            #print(X_train.shape)
            #print(y_train.shape)
            yield X_train, y_train
def simple_model(pretrained_weights = None):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(256, 256, 3)))
    model.add(Convolution2D(100, (3, 3), strides=(2, 2), activation='elu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(200, (3, 3), strides=(2, 2), activation='elu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(300, (3, 3), activation='elu', padding='same'))
    model.add(Convolution2D(300, (3, 3), activation='elu',  padding='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(2, (1, 1))) # this is called upscore layer for some reason?
    model.add(Conv2DTranspose(2, (31, 31), strides=(16, 16), activation='softmax', padding='same'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        
    return model
def predict_batch_from_model(patches, model):
    """Predict which pixels are tumor.
    
    input: patch: `batch_size`x256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    predictions = model.predict(patches)
    predictions = predictions[:, :, :, 1]
    return predictions
def predict_from_model(patch, model):
    """Predict which pixels are tumor.
    
    input: patch: 256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    
    prediction = model.predict(patch.reshape(1, 256, 256, 3))
    prediction = prediction[:, :, :, 1].reshape(256, 256)
    return prediction



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

slide_4_list_1 = [[102,104,29,44],[144,55,30,18],[54,65,21,36],[139,82,1,49],[73,108,7,23],[107,117,24,52],[106,103,27,13]
               ,[105,151,15,2],[75,100,41,9],[156,113,32,37],[150,88,39,10],[84,122,5,50],[93,118,53,47],[87,78,45,34],[116,98,48,46],
                [72,131,22,42],[125,56,40,40]]
slide_4_list_2 = [[109,58,14,28],[101,69,11,43],[94,74,3,20],[64,140,17,16],[92,154,8,26],[99,60,0,33],[86,146,25,19],[68,112,38,51],
                 [71,136,31,4],[59,91,12,6]]
slide_4_list_3 = [[143,132,124,85],[95,120,81,77],[97,96,110,83],[152,128,149,155],[153,111,57,138],[134,135,114,76],
                  [123,90,121,61],[147,148,119,142],[66,137,63,80],[70,79,115,133],[129,141,127,145]]
slide_4_test = [[55,55,0,0]]

columns = ['is_tissue','slide_path','is_tumor','is_all_tumor','tile_loc']

BATCH_SIZE = 32
N_EPOCHS = 5

for i in range(len(slide_4_list_1)):
    # [1] dataset , 2 pos, 2 neg, mean ratio = 3:1
    print(i,'th training\n')
    print(slide_4_list_1[i][0])
    print(slide_4_list_1[i][1]) 
    print(slide_4_list_1[i][2])
    print(slide_4_list_1[i][3])

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
    if NUM_SAMPLES > 20000:
        NUM_SAMPLES = 20000
    
    samples = four_samples.sample(NUM_SAMPLES, random_state=42)
    samples.reset_index(drop=True, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in split.split(samples, samples["is_tumor"]):
            train_samples = samples.loc[train_index]
            validation_samples = samples.loc[test_index]

    train_generator = gen_imgs(four_image_path,four_mask_path,train_samples, BATCH_SIZE)
    validation_generator = gen_imgs(four_image_path,four_mask_path,validation_samples, BATCH_SIZE)
    
    train_start_time = datetime.now()
    history = model.fit_generator(train_generator, np.ceil(len(train_samples) / BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=np.ceil(len(validation_samples) / BATCH_SIZE),
        epochs=N_EPOCHS)

    train_end_time = datetime.now()
    print("Model training time: %.1f minutes" % ((train_end_time - train_start_time).seconds / 60,))
    # split
    # data gen : all_image_path, all_mask_path


path = '/data/model'
model.save(path+'/u_1.h5')
print('model save completed')

