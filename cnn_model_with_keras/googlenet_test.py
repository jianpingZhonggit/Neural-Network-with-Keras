# coding=utf-8
from cnn_model.googlenet_model import check_print
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
folder_list = os.listdir('../dataset/train/')
folder_list = sorted(folder_list)
for img_folder in folder_list:
	img_list = os.listdir('../dataset/train/'+img_folder)
	for i in range(int(len(img_list))):
		number = number + 1
images = np.ones((number, width, height, channels))
labels = np.ones(number, dtype=int)
print(images.shape)
print(labels.shape)
index = 0
label_dict = {"animal": 1, "flower": 2, "guitar": 3, "houses": 4, "plane": 5}
for img_folder in folder_list:
	img_list = os.listdir('../dataset/train/'+img_folder)
	img_list = sorted(img_list)
	for i in range(int(len(img_list))):
		image_name = '../dataset/train/'+img_folder+'/'+img_list[i]
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=30)
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
model = check_print('./google_model_6.h5')
model.fit(X_train, y_train, batch_size=64, epochs=200)
model.save_weights('./google_model_6.h5')
# evaluate model
# loss, accuracy = model.evaluate(X_test, y_test)
# print('\n test loss:', loss)
# print('\n test accuracy:', accuracy)
x = np.ones((50, width, height, channels))
# model.load_weights('./cnn_model.h5')
index = 0
right = []
folder_list = os.listdir('../dataset/val/')
for folder in folder_list:
	img_list = os.listdir('../dataset/val/'+folder)
	for i in range(10):
		x[index] = read_image('../dataset/val/'+folder+'/'+img_list[i])
		index = index + 1
		right.append(label_dict[folder])
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

