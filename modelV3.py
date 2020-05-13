from keras.layers import Input, GlobalAveragePooling2D, Reshape, Conv2D, BatchNormalization, ReLU, Concatenate, Lambda, Add
from keras.models import Model
from backbones import ResNet50
import tensorflow as tf


def deeplabv3(input_tensor=None, input_shape=(512,512,3), classes=21,
              backbone='resnet', multi_grid=(1,2,1), atrous_rates=(6,12,18), output_stride=8):
    if input_tensor is None:
        inpt = Input(input_shape)
    else:
        inpt = input_tensor
        input_shape = inpt._keras_shape[1:]

    # feature extractor: output_stride=16, channels=512
    if backbone == 'resnet':
        feature_extractor = ResNet50(output_stride)
        x = feature_extractor(inpt)

    # AS block4
    x = Atrous_Conv_BN(x, 512, rate=multi_grid)

    # ASPP: [b0: 1x1 conv, b1-3: atrous conv, b4: image pooling]
    x = ASPP(x, atrous_rates, output_stride)

    # decoder
    shortcut = []
    while output_stride>=8:
        # fusion & upSamp
        if shortcut:
            x = Add()([x, shortcut[-1]])
        h, w = x._keras_shape[1:3]
        x = Lambda(lambda x: tf.image.resize_bilinear(x, [h,w]))(x)
        shortcut.append(x)
        output_stride /= 2

    if shortcut:
        x = Add()([x, shortcut[-1]])
    x = Lambda(lambda x: tf.image.resize_bilinear(x, input_shape[:2]))(x)

    # head
    x = Conv2D(classes, 1, strides=1, padding='same', activation='sigmoid')(x)

    model = Model(inpt, x)

    return model


def Atrous_Conv_BN(x, filters, rate):
    if isinstance(rate, tuple):
        # MG block
        for r in rate:
            x = Conv2D(filters, kernel_size=3, strides=1, padding='same', dilation_rate=r)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
    else:
        # ASPP path
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same', dilation_rate=rate)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    return x


def ASPP(x, atrous_rates, output_stride):
    if output_stride==8:
        atrous_rates = [i*2 for i in atrous_rates]
    h, w = x._keras_shape[1:3]
    b0 = Conv2D(256, kernel_size=1, strides=1, padding='same')(x)
    b0 = BatchNormalization()(b0)
    b0 = ReLU()(b0)

    b1 = Atrous_Conv_BN(x, 256, rate=atrous_rates[0])
    b2 = Atrous_Conv_BN(x, 256, rate=atrous_rates[1])
    b3 = Atrous_Conv_BN(x, 256, rate=atrous_rates[2])

    b4 = GlobalAveragePooling2D()(x)
    b4 = Reshape((1,1,x._keras_shape[-1]))(b4)
    b4 = Conv2D(256, kernel_size=1, strides=1, padding='same')(b4)
    b4 = Lambda(lambda x: tf.image.resize_bilinear(x, [h,w]))(b4)

    x = Concatenate(axis=-1)([b0, b1, b2, b3, b4])
    x = Conv2D(256, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


if __name__ == '__main__':

    model = deeplabv3(input_tensor=None, input_shape=(512,512,3), classes=21)
    model.summary()



