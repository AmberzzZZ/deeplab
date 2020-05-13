from keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, add, Input


def ResNet50(output_stride):
    # output stride=16: 3 blocks, channels=512
    # output stride=8
    n_blocks = [3,4,6,3]

    def model(input_tensor):
        x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        if output_stride==8:
            x = id_block(x, filters=64, n_blocks=n_blocks[0])
        else:
            x = conv_block(x, filters=64, n_blocks=n_blocks[0])
        x = conv_block(x, filters=128, n_blocks=n_blocks[1])
        return x

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


# with bottleneck, with conv_bn in id path, have stride2 in 1st conv
def conv_block(x, filters=64, n_blocks=3):
    inpt = x
    inpt = Conv_BN(x, filters*4, kernel_size=1, strides=2, activation=False)
    for i in range(n_blocks):
        strides = 2 if i==0 else 1
        x = Conv_BN(x, filters, kernel_size=1, strides=strides, activation=True)
        x = Conv_BN(x, filters, kernel_size=3, strides=1, activation=True)
        x = Conv_BN(x, filters*4, kernel_size=1, strides=1, activation=False)
    x = add([x, inpt])
    x = ReLU()(x)
    return x


def Conv_BN(x, filters, kernel_size, strides, activation=True):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    feature_extractor = ResNet50(16)
    x = feature_extractor(Input((224,224,3)))
    print(x)

    feature_extractor = ResNet50(8)
    x = feature_extractor(Input((224,224,3)))
    print(x)


