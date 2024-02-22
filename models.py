import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Add, BatchNormalization, Dense
from tensorflow.keras.models import Model


def residual_block(X_start, filters, reduce=False, res_conv2d=False):
    """
    Residual building block used by ResNet-50
    """
    nb_filters_1, nb_filters_2, nb_filters_3 = filters
    strides_1 = [2, 2] if reduce else [1, 1]

    X = Conv2D(filters=nb_filters_1, kernel_size=[1, 1], strides=strides_1, padding='same')(X_start)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=nb_filters_2, kernel_size=[3, 3], strides=[1, 1], padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=nb_filters_3, kernel_size=[1, 1], strides=[1, 1], padding='same')(X)
    X = BatchNormalization()(X)

    if res_conv2d:
        X_res = Conv2D(filters=nb_filters_3, kernel_size=[1, 1], strides=strides_1, padding='same')(X_start)
        X_res = BatchNormalization()(X_res)
    else:
        X_res = X_start

    X = Add()([X, X_res])
    X = Activation('relu')(X)
    return X


def resnet50(input_shape, nb_classes):
    assert len(input_shape) == 3

    X_input = Input(shape=input_shape)

    # conv1
    X = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding='same')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D([3, 3], strides=[2, 2])(X)

    # conv2_x
    X = residual_block(X, filters=[64, 64, 256], reduce=False, res_conv2d=True)
    X = residual_block(X, filters=[64, 64, 256])
    X = residual_block(X, filters=[64, 64, 256])

    # conv3_x
    X = residual_block(X, filters=[128, 128, 512], reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[128, 128, 512])
    X = residual_block(X, filters=[128, 128, 512])
    X = residual_block(X, filters=[128, 128, 512])

    # conv4_x
    X = residual_block(X, filters=[256, 256, 1024], reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])

    # conv5_x
    X = residual_block(X, filters=[512, 512, 2048], reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[512, 512, 2048])
    X = residual_block(X, filters=[512, 512, 2048])

    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(units=nb_classes, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    return model
