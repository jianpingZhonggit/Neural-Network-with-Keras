# coding=utf-8
from keras import Model, Input
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
import os


def get_alex_model(pre_weight, input_size):
    # inputs
    inputs = Input(input_size)

    # convolution1
    convolution1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(inputs)

    # pooling1
    pooling1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(convolution1)

    # convolution2
    convolution2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(pooling1)

    # pooling2
    pooling2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(convolution2)

    # convolution3
    convolution3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pooling2)

    # convolution4
    convolution4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          activation='relu')(convolution3)

    # convolution5
    convolution5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          activation='relu')(convolution4)

    # pooling5
    pooling5 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(convolution5)

    # fc1
    fc1 = Flatten()(pooling5)

    # fc2 and drop2
    fc2 = Dense(4096, activation='relu')(fc1)
    drop2 = Dropout(0.5)(fc2)

    # fc3 and drop3
    fc3 = Dense(4096, activation='relu')(drop2)
    drop3 = Dropout(0.5)(fc3)

    # outputs
    outputs = Dense(1000, activation='softmax')(drop3)

    # model
    model = Model(inputs=inputs, outputs=outputs)

    # optimizer
    sgd = SGD(lr=1e-5, decay=0.5)
    adam = Adam(lr=1e-6, decay=0.5)

    # load pre_weight
    if os.path.exists(pre_weight):
        model.load_weights(pre_weight)

    # compile model
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # print the model structure
    model.summary()

    return model


model = get_alex_model('./aa.h5', (224, 224, 3))


