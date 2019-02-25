# import os
# import gc

# import keras
# import numpy as np

# from PIL import Image
# from keras import layers, models
# from keras import backend as K
# from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical

# from preprocessing import open_slide, create_tiles
# from preprocessing import create_patches
# from model import create_model

# print('**************************** run train.py ***************************************')
# train_dir = './tiles/train/'
# val_dir = './tiels/val/'

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=90,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     brightness_range =(0.65, 1.),
#     fill_mode='reflect')

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(256, 256), 
#         batch_size=32,
#         class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         val_dir,
#         target_size=(256, 256),
#         batch_size=32,
#         class_mode='binary')

# # binary -> category (batch_size, 256, 256, 2)
# def generator(gen):
#     while 1:
#         train_x, labels = next(gen)
#         train_y = []
#         for label in labels:
#             if label == 0:
#                 train_y.append(np.zeros((256, 256)))
#             else:
#                 train_y.append(np.ones((256, 256), dtype='float'))
#         train_y = np.array(train_y)
#         train_y = to_categorical(train_y, num_classes=2)
#         yield train_x, train_y


# train_gen = generator(train_generator)
# val_gen = generator(validation_generator)

# print('============ Create model for Training ==============')
# model = create_model(pretrained_weights='./incept-unet.h5')
# print('============ Model Loaded for Training ==============')

# batch_size = 32
# n_epochs = 10
# val_steps = 45 * 102 // batch_size
# model.fit_generator(train_gen,
#           steps_per_epoch=300, 
#           epochs=n_epochs,
#           validation_data=val_gen,
#           validation_steps=val_steps,
#           verbose=1)

# print('========= Model saving... =======')
# model.save_weights('/data/model/incept-unet.h5')
# print('========= Model saved!!!! ========')
# print('********************** Train finished **********************')

print('test for docker!!')