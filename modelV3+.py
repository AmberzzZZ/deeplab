from modelV3 import ASPP
from backbones import Xception, Conv_BN
from keras.layers import Input, concatenate, Lambda, Conv2D
from keras.models import Model
import tensorflow as tf


def deeplabV3_plus(input_tensor=None, input_shape=(512,512,3), classes=21,
                   backbone='xception', atrous_rates=(6,12,18), output_stride=16):
    if input_tensor is None:
        inpt = Input(input_shape)
    else:
        inpt = input_tensor
        input_shape = inpt._keras_shape[1:]

    # feature extractor
    if backbone == 'xception':
        feature_extractor = Xception()
        x, shortcut = feature_extractor(inpt)

    # ASPP
    x = ASPP(x, atrous_rates, output_stride)

    # decoder
    # x4 & concat
    h, w = x._keras_shape[1:3]
    x = Lambda(lambda x: tf.image.resize_bilinear(x, [h*4,w*4]))(x)
    shortcut = Conv_BN(shortcut, 48, kernel_size=1, strides=1, activation=False)
    x = concatenate([x, shortcut])
    x = Conv_BN(x, 256, kernel_size=3, strides=1, activation=True)
    x = Conv_BN(x, 256, kernel_size=3, strides=1, activation=True)
    # straight upSamp to origin resolu
    x = Lambda(lambda x: tf.image.resize_bilinear(x, input_shape[:2]))(x)

    # head
    x = Conv2D(classes, 1, strides=1, padding='same', activation='sigmoid')(x)

    model = Model(inpt, x)

    return model


if __name__ == '__main__':

    model = deeplabV3_plus(input_tensor=None, input_shape=(512,512,3), classes=21)
    model.summary()







