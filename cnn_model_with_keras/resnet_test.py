# coding=utf-8
from cnn_model.resnet_model import ResNet50
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
ktf.set_session(sess)
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
folder_list = os.listdir('../flower/')
folder_list = sorted(folder_list)
for img_folder in folder_list:
	img_list = os.listdir('../flower/'+img_folder)
	for i in range(int(len(img_list)/2)):
		number = number + 1
images = np.ones((number, width, height, channels))
labels = np.ones(number, dtype=int)
print(images.shape)
print(labels.shape)
index = 0
for img_folder in folder_list:
	img_list = os.listdir('../flower/'+img_folder)
	img_list = sorted(img_list)
	for i in range(int(len(img_list)/2)):
		image_name = '../flower/'+img_folder+'/'+img_list[i]
		images[index] = read_image(image_name)
		labels[index] = int(img_folder)
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

y_train = np_utils.to_categorical(y_train, num_classes=103)
y_test = np_utils.to_categorical(y_test, num_classes=103)
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
model = ResNet50('./res_model.h5')
model.fit(X_train, y_train, batch_size=16, epochs=40)
# evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('\n test loss:', loss)
print('\n test accuracy:', accuracy)
model.save_weights('./res_model.h5')
