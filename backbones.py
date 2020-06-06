from keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, add, Input, DepthwiseConv2D
from keras.models import Model


def ResNet50(output_stride):
    # output stride=16: 3 blocks, channels=512
    # output stride=8
    n_blocks = [3,4,6,3]

    def model(input_tensor):
        x = Conv_BN(input_tensor, 64, kernel_size=7, strides=2)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        if output_stride==8:
            x = id_block(x, filters=64, n_blocks=n_blocks[0])
        else:
            x = conv_block(x, filters=64, strides=2, n_blocks=n_blocks[0])
        x = conv_block(x, filters=128, strides=2, n_blocks=n_blocks[1])
        return x

    return model


def Xception():
    def model(input_tensor):
        # entry flow
        x = Conv_BN(input_tensor, 32, kernel_size=3, strides=2)
        x = Conv_BN(x, 64, kernel_size=3, strides=1)
        x = sep_conv_block(x, filters=128, strides=2)
        x = sep_conv_block(x, filters=256, strides=2)
        shortcut = x
        x = sep_conv_block(x, filters=728, strides=2)
        # middle flow
        for i in range(16):
            x = sep_id_block(x, filters=728)
        # exit flow
        x = sep_conv_block(x, filters=(728,1024,1024), strides=2)
        x = Sep_Conv_BN(x, filters=1536, strides=1)
        x = Sep_Conv_BN(x, filters=1536, strides=1)
        x = Sep_Conv_BN(x, filters=2048, strides=1)
        return x, shortcut
    return model


# without bottleneck, without conv_bn in id path, no stride2
def id_block(x, filters=64, n_blocks=3):
    inpt = x
    for i in range(n_blocks):
        x = Conv_BN(x, filters, kernel_size=1, strides=1, activation=True)
        x = Conv_BN(x, filters, kernel_size=3, strides=1, activation=True)
        x = Conv_BN(x, filters, kernel_size=1, strides=1, activation=False)
    x = add([x, inpt])
    x = ReLU()(x)
    return x


# with bottleneck, with conv_bn in id path, may have stride2 in 1st conv
def conv_block(x, filters=64, strides=1, n_blocks=3):
    inpt = x
    inpt = Conv_BN(x, filters*4, kernel_size=1, strides=strides, activation=False)
    for i in range(n_blocks):
        stride = strides if i==0 else 1
        x = Conv_BN(x, filters, kernel_size=1, strides=stride, activation=True)
        x = Conv_BN(x, filters, kernel_size=3, strides=1, activation=True)
        x = Conv_BN(x, filters*4, kernel_size=1, strides=1, activation=False)
    x = add([x, inpt])
    x = ReLU()(x)
    return x


def Conv_BN(x, filters, kernel_size, strides, activation=True, dilation_rate=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


# with conv_bn in id path, may have stride2 in last conv
def sep_conv_block(x, filters, strides):
    if isinstance(filters, int):
        inpt = x
        inpt = Conv_BN(inpt, filters, kernel_size=1, strides=strides, activation=False)

        x = Sep_Conv_BN(x, filters, strides=1, activation=True)
        x = Sep_Conv_BN(x, filters, strides=1, activation=True)
        x = Sep_Conv_BN(x, filters, strides=strides, activation=False)
    else:
        inpt = x
        inpt = Conv_BN(inpt, filters[-1], kernel_size=1, strides=strides, activation=False)

        for idx, f in enumerate(filters):
            activation = True if idx < len(filters)-1 else False
            stride = 1 if idx < len(filters)-1 else strides
            x = Sep_Conv_BN(x, filters=f, strides=stride, activation=activation)

    x = add([inpt, x])
    x = ReLU()(x)

    return x


# without conv_bn in id path, in-out channel keep the same(without strides)
def sep_id_block(x, filters):
    inpt = x

    x = Sep_Conv_BN(x, filters, strides=1, activation=True)
    x = Sep_Conv_BN(x, filters, strides=1, activation=True)
    x = Sep_Conv_BN(x, filters, strides=1, activation=False)

    x = add([inpt, x])
    x = ReLU()(x)

    return x


def Sep_Conv_BN(x, filters, strides, activation=True):
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv_BN(x, filters, kernel_size=1, strides=strides, activation=True, dilation_rate=1)
    return x


if __name__ == '__main__':

    # feature_extractor = ResNet50(16)
    # x = feature_extractor(Input((224,224,3)))
    # print(x)

    # feature_extractor = ResNet50(8)
    # x = feature_extractor(Input((224,224,3)))
    # print(x)

    inpt = Input((224,224,3))

    # feature_extractor = Xception()
    # model = Model(inpt, feature_extractor(inpt)[0])
    # model.summary()

    feature_extractor = ResNet50(16)
    model = Model(inpt, feature_extractor(inpt))
    model.summary()



