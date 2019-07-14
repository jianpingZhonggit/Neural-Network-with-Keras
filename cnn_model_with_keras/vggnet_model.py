from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
import os


def get_model(pre_weight, input_size):
    inputs = Input(input_size)

    # convolution1
    convolution1_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    convolution1_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(convolution1_1)

    # pooling1
    pooling1 = MaxPool2D((2, 2), strides=(2, 2))(convolution1_2)

    # convolution2
    convolution2_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(pooling1)
    convolution2_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(convolution2_1)

    # pooling2
    pooling2 = MaxPool2D((2, 2), strides=(2, 2))(convolution2_2)

    # convolution3
    convolution3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(pooling2)
    convolution3_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(convolution3_1)
    convolution3_3 = Conv2D(256, (3, 3), padding='same', activation='relu')(convolution3_2)

    # pooling3
    pooling3 = MaxPool2D((2, 2), strides=(2, 2))(convolution3_3)

    # convolution4
    convolution4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(pooling3)
    convolution4_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(convolution4_1)
    convolution4_3 = Conv2D(512, (3, 3), padding='same', activation='relu')(convolution4_2)

    # pooling4
    pooling4 = MaxPool2D((2, 2), strides=(2, 2))(convolution4_3)

    # convolution5
    convolution5_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(pooling4)
    convolution5_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(convolution5_1)
    convolution5_3 = Conv2D(512, (3, 3), padding='same', activation='relu')(convolution5_2)

    # pooling5
    pooling5 = MaxPool2D((2, 2), strides=(2, 2))(convolution5_3)

    # fc1
    fc1 = Flatten()(pooling5)

    # fc2
    fc2 = Dense(4096, activation='relu')(fc1)

    # fc3
    fc3 = Dense(4096, activation='relu')(fc2)

    # output
    output = Dense(103, activation='softmax')(fc3)

    model = Model(inputs=inputs, outputs=output)

    model.summary()
    if os.path.exists(pre_weight):
        print('exist')
        model.load_weights(pre_weight)
    adam = Adam(lr=1e-4, decay=0.5)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
