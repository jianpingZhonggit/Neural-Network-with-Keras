from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Softmax, BatchNormalization, ReLU
from keras.optimizers import Adam


def get_model(input_size):
    inputs = Input(input_size)

    # convolution1
    convolution1 = Conv2D(32, (3, 3), padding='same')(inputs)
    b1 = BatchNormalization()(convolution1)
    r1 = ReLU()(b1)
    # pooling1
    pooling1 = MaxPool2D((2, 2), 2)(r1)

    # convolution2
    convolution2 = Conv2D(64, (3, 3), padding='same')(pooling1)
    b2 = BatchNormalization()(convolution2)
    r2 = ReLU()(b2)
    # pooling2
    pooling2 = MaxPool2D((2, 2), 2)(r2)

    # convolution3
    convolution3 = Conv2D(128, (3, 3), padding='same')(pooling2)
    b3 = BatchNormalization()(convolution3)
    r3 = ReLU()(b3)
    # convolution4
    convolution4 = Conv2D(64, (1, 1), padding='same')(r3)
    b4 = BatchNormalization()(convolution4)
    r4 = ReLU()(b4)
    # convolution5
    convolution5 = Conv2D(128, (3, 3), padding='same')(r4)
    b5 = BatchNormalization()(convolution5)
    r5 = ReLU()(b5)
    # pooling3
    pooling3 = MaxPool2D((2, 2), 2)(r5)

    # convolution6
    convolution6 = Conv2D(256, (3, 3), padding='same')(pooling3)
    b6 = BatchNormalization()(convolution6)
    r6 = ReLU()(b6)
    # convolution7
    convolution7 = Conv2D(128, (1, 1), padding='same')(r6)
    b7 = BatchNormalization()(convolution7)
    r7 = ReLU()(b7)
    # convolution8
    convolution8 = Conv2D(256, (3, 3), padding='same')(r7)
    b8 = BatchNormalization()(convolution8)
    r8 = ReLU()(b8)
    # pooling4
    pooling4 = MaxPool2D((2, 2), 2)(r8)

    # convolution9
    convolution9 = Conv2D(512, (3, 3), padding='same')(pooling4)
    b9 = BatchNormalization()(convolution9)
    r9 = ReLU()(b9)
    # convolution10
    convolution10 = Conv2D(256, (1, 1), padding='same')(r9)
    b10 = BatchNormalization()(convolution10)
    r10 = ReLU()(b10)
    # convolution11
    convolution11 = Conv2D(512, (3, 3), padding='same')(r10)
    b11 = BatchNormalization()(convolution11)
    r11 = ReLU()(b11)
    # convolution12
    convolution12 = Conv2D(256, (1, 1), padding='same')(r11)
    b12 = BatchNormalization()(convolution12)
    r12 = ReLU()(b12)
    # convolution13
    convolution13 = Conv2D(512, (3, 3), padding='same')(r12)
    b13 = BatchNormalization()(convolution13)
    r13 = ReLU()(b13)
    # pooling5
    pooling5 = MaxPool2D((2, 2), 2)(r13)

    # convolution14
    convolution14 = Conv2D(1024, (3, 3), padding='same')(pooling5)
    b14 = BatchNormalization()(convolution14)
    r14 = ReLU()(b14)
    # convolution15
    convolution15 = Conv2D(512, (1, 1), padding='same')(r14)
    b15 = BatchNormalization()(convolution15)
    r15 = ReLU()(b15)
    # convolution16
    convolution16 = Conv2D(1024, (3, 3), padding='same')(r15)
    b16 = BatchNormalization()(convolution16)
    r16 = ReLU()(b16)
    # convolution17
    convolution17 = Conv2D(512, (1, 1), padding='same')(r16)
    b17 = BatchNormalization()(convolution17)
    r17 = ReLU()(b17)
    # convolution18
    convolution18 = Conv2D(1024, (3, 3), padding='same')(r17)
    b18 = BatchNormalization()(convolution18)
    r18 = ReLU()(b18)
    # convolution19
    convolution19 = Conv2D(102, (1, 1), padding='same')(r18)
    b19 = BatchNormalization()(convolution19)
    r19 = ReLU()(b19)

    # average
    average = GlobalAveragePooling2D()(r19)

    # outputs
    outputs = Softmax()(average)

    adam = Adam(lr=1e-4, decay=1e-2)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    return model


# get_model((224, 224, 3))

























