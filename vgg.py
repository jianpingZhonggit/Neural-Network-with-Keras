from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import cv2
np.random.seed(1337)
width = 224
height = 224
channels = 3


def read_image(img_name):
	img = cv2.imread(img_name)
	img = cv2.resize(img, (width, height))
	data = np.array(img)
	return data


# for fn in os.listdir('./images'):
#     if fn.endswith('.png'):
#         fd = os.path.join('./images', fn)
#         images.append(read_image(fd))
number = 0
label_dict = {"animal": 1, "flower": 2, "guitar": 3, "houses": 4, "plane": 5}
folder_list = os.listdir('./dataset/train/')
folder_list = sorted(folder_list)
for img_folder in folder_list:
	img_list = os.listdir('./dataset/train/'+img_folder)
	for i in range(int(len(img_list))):
		number = number + 1
images = np.ones((number, width, height, channels))
labels = np.ones(number, dtype=int)
print(images.shape)
print(labels.shape)
index = 0
for img_folder in folder_list:
	img_list = os.listdir('./dataset/train/'+img_folder)
	img_list = sorted(img_list)
	for i in range(int(len(img_list))):
		image_name = './dataset/train/'+img_folder+'/'+img_list[i]
		images[index] = read_image(image_name)
		labels[index] = label_dict[img_folder]
		index = index + 1
print('load success!')
X = images
print(X.shape)

# y = np.array([labels])
y = labels
# print(labels[155])
# print(labels)
# y = np.loadtxt('out.txt')
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print("Changing format......")

y_train = np_utils.to_categorical(y_train, num_classes=6)
y_test = np_utils.to_categorical(y_test, num_classes=6)
print(y.shape)
print("*" * 30)
print(X_train[187])
print("*" * 30)
print(y_train[187])
print("*" * 30)
# print(y_train[10])
# print(len(y_train[100]))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("Changing succeeded!")
# os.system("pause")
# #########################################################################
model = Sequential()
# #1:64
model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

# #2:128
model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2), padding='same'))

# #3:256
model.add(Convolution2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2), padding='same'))

# #4:512
model.add(Convolution2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2, padding='same'))

# #5:512
model.add(Convolution2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2), padding='same'))

# ####FC
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))
model.summary()
# ########################
adam = Adam(lr=1e-4, decay=0.5)
#
# ########################
if os.path.exists("./cnn_model_6.h5"):
	print("exist")
	model.load_weights("./cnn_model_6.h5")
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# if os.path.exists('./cnn_model.h5'):
# 	model.load_weights('./cnn_model.h5')
#
check_point = ModelCheckpoint('./cnn_model_6.h5', monitor='loss', save_best_only=True)
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=[check_point])

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
x = np.ones((50, width, height, channels))
# model.load_weights('./cnn_model.h5')
index = 0
right = []
folder_list = os.listdir('./dataset/val/')
for folder in folder_list:
	img_list = os.listdir('./dataset/val/'+folder)
	for i in range(10):
		x[index] = read_image('./dataset/val/'+folder+'/'+img_list[i])
		index = index + 1
		right.append()
y = model.predict(x, batch_size=16)
res = []
for y_ in y:
	res.append(np.argmax(y_))
print(y.shape)
print(res)
print(right)
count = 0
print(len(right))
for i in range(50):
	if res[i] == right[i]:
		count = count + 1
print(count)
print(count/50)
#
# model.save('cnn_model.h5')
# model.save_weights('./cnn_model.h5')
# print('\nSuccessfully saved as cnn_model.h5')
