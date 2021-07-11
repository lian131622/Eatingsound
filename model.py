from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers, Model, optimizers


def conv2D(input_dim):
    model = Sequential()
    model.add(Convolution2D(64, (4, 28), padding="same", input_shape=input_dim, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 5)))
    model.add(Convolution2D(128, (2, 14), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 5)))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def conv1D(input_dim):
    model = Sequential()
    model.add(Conv1D(16, 3, padding="same", activation="relu", input_shape=input_dim))  # 卷积层
    model.add(Conv1D(16, 3, padding="same", activation="relu"))  # 卷积层
    model.add(Conv1D(16, 3, padding="same", activation="relu"))  # 卷积层
    model.add(BatchNormalization())  # BN层
    model.add(Dropout(0.52, seed=66))
    model.add(Flatten())  # 展开
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.52, seed=66))
    model.add(Dense(20, activation="softmax"))  # 输出层：20个units输出20个类的概率
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def model11(input_dim):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding="same", activation="tanh", input_shape=input_dim))  # 卷积层
    model.add(MaxPool2D(pool_size=(3, 5)))  # 最大池化
    model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))  # 卷积层
    model.add(MaxPool2D(pool_size=(3, 5)))  # 最大池化层

    model.add(Dropout(0.6))
    model.add(Flatten())  # 展开
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(20, activation="softmax"))  # 输出层：20个units输出20个类的概率

    # 编译模型，设置损失函数，优化方法以及评价标准
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def __causal_gated_conv1D(x=None, filters=16, length=6, strides=1):
    def causal_gated_conv1D(x, filters, length, strides):
        x_in_1 = layers.Conv1D(filters=filters // 2,
                               kernel_size=length,
                               dilation_rate=strides,  # it's correct, use this instead strides for shape matching
                               strides=1,
                               padding="causal")(x)
        x_sigmoid = layers.Activation(activation="sigmoid")(x_in_1)

        x_in_2 = layers.Conv1D(filters=filters // 2,
                               kernel_size=length,
                               dilation_rate=strides,  # it's correct, use this instead strides for shape matching
                               strides=1,
                               padding="causal")(x)
        x_tanh = layers.Activation(activation="tanh")(x_in_2)

        x_out = layers.Multiply()([x_sigmoid, x_tanh])

        return x_out

    if x is None:
        return lambda _x: causal_gated_conv1D(x=_x, filters=filters, length=length, strides=strides)
    else:
        return causal_gated_conv1D(x=x, filters=filters, length=length, strides=strides)


def SwishNet(input_shape, classes, width_multiply=1):
    _x_in = layers.Input(shape=input_shape)

    # 1 block
    _x_up = __causal_gated_conv1D(filters=16 * width_multiply, length=3)(_x_in)
    _x_down = __causal_gated_conv1D(filters=16 * width_multiply, length=6)(_x_in)
    _x = layers.Concatenate()([_x_up, _x_down])

    # 2 block
    _x_up = __causal_gated_conv1D(filters=8 * width_multiply, length=3)(_x)
    _x_down = __causal_gated_conv1D(filters=8 * width_multiply, length=6)(_x)
    _x = layers.Concatenate()([_x_up, _x_down])

    # 3 block
    _x_up = __causal_gated_conv1D(filters=8 * width_multiply, length=3)(_x)
    _x_down = __causal_gated_conv1D(filters=8 * width_multiply, length=6)(_x)
    _x_concat = layers.Concatenate()([_x_up, _x_down])

    _x = layers.Add()([_x, _x_concat])

    # 4 block
    _x_loop1 = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=3)(_x)
    _x = layers.Add()([_x, _x_loop1])

    # 5 block
    _x_loop2 = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=2)(_x)
    _x = layers.Add()([_x, _x_loop2])

    # 6 block
    _x_loop3 = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=2)(_x)
    _x = layers.Add()([_x, _x_loop3])

    # 7 block
    _x_forward = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=2)(_x)

    # 8 block
    _x_loop4 = __causal_gated_conv1D(filters=32 * width_multiply, length=3, strides=2)(_x)

    # output
    _x = layers.Concatenate()([_x_loop2, _x_loop3, _x_forward, _x_loop4])
    _x = layers.Conv1D(filters=classes, kernel_size=1)(_x)
    _x = layers.GlobalAveragePooling1D()(_x)
    _x = layers.Activation("softmax")(_x)

    model = Model(inputs=_x_in, outputs=_x)
    # adam = optimizers.Adam(learning_rate= 0.005)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
