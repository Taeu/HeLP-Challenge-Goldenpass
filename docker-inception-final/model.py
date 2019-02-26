# inception
# imagenet_utils에 있는 함수들
"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import imagenet_utils
import keras
_KERAS_BACKEND = keras.backend
_KERAS_LAYERS = keras.layers
_KERAS_MODELS = keras.models
_KERAS_UTILS = keras.utils

import os.path as osp
import openslide
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 이부분 python 에서는 뺴주기

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


import math
from PIL import Image
from xml.etree.ElementTree import ElementTree, Element, SubElement
from io import BytesIO
import skimage.io as io

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import keras.backend.tensorflow_backend as K
from sklearn import metrics
from keras_applications import imagenet_utils
# network

from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import *


down_para = 2
PATCH_SIZE = 256
NO_EPOCHS = 10

def set_keras_submodules(backend=None,
                         layers=None,
                         models=None,
                         utils=None,
                         engine=None):
    # Deprecated, will be removed in the future.
    global _KERAS_BACKEND
    global _KERAS_LAYERS
    global _KERAS_MODELS
    global _KERAS_UTILS
    _KERAS_BACKEND = backend
    _KERAS_LAYERS = layers
    _KERAS_MODELS = models
    _KERAS_UTILS = utils
def get_keras_submodule(name):
    # Deprecated, will be removed in the future.
    if name not in {'backend', 'layers', 'models', 'utils'}:
        raise ImportError(
            'Can only retrieve one of "backend", '
            '"layers", "models", or "utils". '
            'Requested: %s' % name)
    if _KERAS_BACKEND is None:
        raise ImportError('You need to first `import keras` '
                          'in order to use `keras_applications`. '
                          'For instance, you can do:\n\n'
                          '```\n'
                          'import keras\n'
                          'from keras_applications import vgg16\n'
                          '```\n\n'
                          'Or, preferably, this equivalent formulation:\n\n'
                          '```\n'
                          'from keras import applications\n'
                          '```\n')
    if name == 'backend':
        return _KERAS_BACKEND
    elif name == 'layers':
        return _KERAS_LAYERS
    elif name == 'models':
        return _KERAS_MODELS
    elif name == 'utils':
        return _KERAS_UTILS
def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils
def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

__version__ = '1.0.7'

"""Inception V3 model for Keras.
Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).
# Reference
- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567)
"""
import os

WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# 이부분도
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x
def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                down_para = down_para,
                **kwargs):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)



    # Determine proper input shape
    input_shape = imagenet_utils._obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=75,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32//down_para, 3, 3, strides=(2, 2), padding='same')
    x = conv2d_bn(x, 32//down_para, 3, 3, padding='same')
    x = conv2d_bn(x, 64//down_para, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80//down_para, 1, 1, padding='same')
    x = conv2d_bn(x, 192//down_para, 3, 3, padding='same')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64//down_para, 1, 1)

    branch5x5 = conv2d_bn(x, 48//down_para, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64//down_para, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64//down_para, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//down_para, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//down_para, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32//down_para, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64//down_para, 1, 1)

    branch5x5 = conv2d_bn(x, 48//down_para, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64//down_para, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64//down_para, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//down_para, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//down_para, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64//down_para, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64//down_para, 1, 1)

    branch5x5 = conv2d_bn(x, 48//down_para, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64//down_para, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64//down_para, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//down_para, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//down_para, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64//down_para, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384//down_para, 3, 3, strides=(2, 2), padding='same')

    branch3x3dbl = conv2d_bn(x, 64//down_para, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//down_para, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96//down_para, 3, 3, strides=(2, 2), padding='same')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192//down_para, 1, 1)

    branch7x7 = conv2d_bn(x, 128//down_para, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128//down_para, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192//down_para, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128//down_para, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128//down_para, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128//down_para, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128//down_para, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192//down_para, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192//down_para, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192//down_para, 1, 1)

        branch7x7 = conv2d_bn(x, 160//down_para, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160//down_para, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192//down_para, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160//down_para, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160//down_para, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160//down_para, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160//down_para, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192//down_para, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192//down_para, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192//down_para, 1, 1)

    branch7x7 = conv2d_bn(x, 192//down_para, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192//down_para, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192//down_para, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192//down_para, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192//down_para, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192//down_para, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192//down_para, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192//down_para, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192//down_para, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # 여기서부터 upsampling 해보자, 14,14,192
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x))
    #up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(2, 1, activation = 'softmax')(conv9)

    model = Model(input = img_input, output = conv10)
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-2), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if weights is not None:
        model.load_weights(weights)

    return model 

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='inception_v3')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


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
file_handles=[]

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
    
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
        
    return model

def unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    inputs_norm = Lambda(lambda x: x /255.0 - 0.5)(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_norm)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.2)(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(2, 1, activation = 'softmax')(conv9)

    model = Model(input = inputs, output = conv10)
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



