from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def create_model(patch_size=256, pretrained_weights=False):
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

    outputs = layers.Conv2D(1, 1, activation='sigmoid', 
                            kernel_initializer='he_normal')(conv9)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adamax(),
                  loss='binary_crossentropy', 
                  metrics=['acc'])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model