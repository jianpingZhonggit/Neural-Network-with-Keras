# import tensorflow as tf
# Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
# with tf.Session() as sess:
#     print(sess.run(c))
# print(c)
# right = []
# for i in range(1, 103):
#     if i % 2 == 0:
#         right.append(int(i/2))
#     else:
#         right.append(int(i/2)+1)
# print(right)
from vgg16_model import get_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import os
import numpy as np
import cv2
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
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
folder_list = os.listdir('./flower/')
folder_list = sorted(folder_list)
for img_folder in folder_list:
    img_list = os.listdir('./flower/'+img_folder)
    for i in range(int(len(img_list)*4/5)):
        number = number + 1
images = np.ones((number, width, height, channels))
labels = np.ones(number, dtype=int)
print(images.shape)
print(labels.shape)
index = 0
for img_folder in folder_list:
    img_list = os.listdir('./flower/'+img_folder)
    img_list = sorted(img_list)
    for i in range(int(len(img_list)*4/5)):
        image_name = './flower/'+img_folder+'/'+img_list[i]
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
model = get_model('./vgg_model.h5')
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16,
						 write_grads=True, write_graph=True, write_images=True,
						 embeddings_freq=0, embeddings_layer_names=None,
						 embeddings_metadata=None)
model.fit(X_train, y_train, batch_size=64, epochs=200, callbacks=[tbCallBack])
loss, accuracy = model.evaluate(X_test, y_test, batch_size=64)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
x = np.ones((204, width, height, channels))
# model.load_weights('./cnn_model.h5')
index = 0
right = []
folder_list = os.listdir('./flower/')
for folder in folder_list:
    img_list = os.listdir('./flower/'+folder)
    for i in range(2):
        x[index] = read_image('./flower/'+folder+'/'+img_list[i])
        index = index + 1
        right.append(int(folder))
y = model.predict(x, batch_size=16)
res = []
for y_ in y:
    res.append(np.argmax(y_))
print(y.shape)
print(res)
print(right)
count = 0
print(len(right))
for i in range(204):
    if res[i] == right[i]:
        count = count + 1
print(count)
print(count/204)
#
# model.save('cnn_model.h5')
model.save_weights('./vgg_model.h5')
print('\nSuccessfully saved as vgg_model.h5')

