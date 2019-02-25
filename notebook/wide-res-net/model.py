import keras
from keras import layers, models
from keras import backend as K
from keras import optimizers


def inception_block(inputs):
    branch_a = layers.Conv2D(64//2, 1, padding='same',
                         activation='relu')(inputs)
    branch_a = layers.BatchNormalization()(branch_a)
    
    branch_b = layers.Conv2D(48//2, 1, padding='same',
                             activation='relu')(inputs)
    branch_b = layers.BatchNormalization()(branch_b)
    branch_b = layers.Conv2D(96//2, 3, padding='same',
                             activation='relu')(branch_b)
    branch_b = layers.BatchNormalization()(branch_b)

    branch_c = layers.AveragePooling2D(3, strides=1, padding='same')(inputs)
    branch_c = layers.Conv2D(48//2, 3, padding='same', activation='relu')(branch_c)
    branch_c = layers.BatchNormalization()(branch_c)

    branch_d = layers.Conv2D(48//2, 1, padding='same', activation='relu')(inputs)
    branch_d = layers.BatchNormalization()(branch_d)
    branch_d = layers.Conv2D(48//2, 3, padding='same', activation='relu')(branch_d)
    branch_d = layers.BatchNormalization()(branch_d)

    # Concatenate
    concatenated = layers.concatenate([branch_a, branch_b, 
                                       branch_c, branch_d], axis=-1)
    
    return concatenated


def create_model(pretrained_weights=None):
    inputs = layers.Input(shape=(256, 256, 3), dtype='float32')
    conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D()(conv1)

    conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D()(conv2)

    conv3 = layers.Conv2D(128, 3, padding='same', activation='relu')(pool1)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(128, 3, padding='same', activation='relu')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    incept1 = inception_block(conv3)
    res1 = layers.add([incept1, conv3])
    pool3 = layers.MaxPooling2D()(res1)

    incept2 = inception_block(pool3)
    res2 = layers.add([incept2, pool3])
    pool4 = layers.MaxPooling2D()(res2)

    incept3 = inception_block(pool4)
    res3 = layers.add([incept3, pool4])
    pool5 = layers.MaxPooling2D()(res3)

    incept4 = inception_block(pool5)
    res4 = layers.add([incept4, pool5])
    pool6 = layers.MaxPooling2D()(res4)

    # incept5 = inception_block(pool6)
    # res5 = layers.add([incept5, pool6])
    # pool7 = layers.MaxPooling2D()(res5)

    # Up-Conv layers
    up_conv5 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(pool6)
    up_conv5 = layers.concatenate([up_conv5, pool5])
    conv5 = layers.Conv2D(128, 3, padding='same', activation='relu')(up_conv5)

    up_conv6 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv5)
    up_conv6 = layers.concatenate([up_conv6, pool4])
    conv6 = layers.Conv2D(64, 3, padding='same', activation='relu')(up_conv6)

    up_conv7 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(conv6)
    up_conv7 = layers.concatenate([up_conv7, pool3])
    conv7 = layers.Conv2D(32, 3, padding='same', activation='relu')(up_conv7)

    up_conv8 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(conv7)
    up_conv8 = layers.concatenate([up_conv8, conv3])
    conv8 = layers.Conv2D(16, 3, padding='same', activation='relu')(up_conv8)

    up_conv9 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(conv8)
    up_conv9 = layers.concatenate([up_conv9, conv1])
    conv9 = layers.Conv2D(16, 3, padding='same', activation='relu')(up_conv9)

    outputs = layers.Conv2D(2, 1, activation='softmax')(conv9)

    model = models.Model(inputs, outputs)

    model.compile(optimizer=optimizers.Adamax(),
                  loss='categorical_crossentropy', 
                  metrics=['acc'])


    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model