import os
import numpy as np
import cv2
import tensorflow as tf
from keras.utils import np_utils
from cnn_model_with_keras.darknet19_model import get_model
from keras.models import load_model
from keras.backend import tensorflow_backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
k.set_session(sess)
width = 224
height = 224
channels = 3


def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    # img = np.array(img)
    return img


# 类别文件夹
folder_list = os.listdir("../flower/")
# 统计样本数
number = 0
for folder in folder_list:
    img_list = os.listdir("../flower/"+folder)
    number += len(img_list)

# 读取图片和标签数据
images = np.ones((number, width, height, channels), dtype=np.int)
labels = np.ones(number, dtype=np.int)
index = 0
for folder in folder_list:
    img_list = os.listdir("../flower/"+folder)
    for img_name in img_list:
        images[index] = read_img("../flower/"+folder+"/"+img_name)
        # labels[index] = np.array(int(folder)-1)
        labels[index] = int(folder) - 1
        index = index + 1

x_train = images
y_train = np_utils.to_categorical(labels, num_classes=102)
if os.path.exists("./darknet19_model_with_normalization_activation.h5"):
    model = load_model('./darknet19_model_with_normalization_activation.h5')
else:
    model = get_model((width, height, channels))
model.fit(x_train, y_train, epochs=5, batch_size=32)
model.save("./darknet19_model_with_normalization_activation.h5")
x_test = np.ones((204, width, height, channels))
y_test = []
index = 0
for folder in folder_list:
    img_list = os.listdir("../flower/"+folder+"/")
    for i in range(2):
        x_test[index] = read_img("../flower/"+folder+"/"+img_list[i])
        index = index + 1
        y_test.append(int(folder)-1)
y_predict = model.predict(x_test)
print(y_predict.shape)
predict = []
correct = 0
index = 0
for _ in y_predict:
    predict.append(np.argmax(_))
    if np.argmax(_) == y_test[index]:
        correct = correct + 1
    index = index + 1
print("accuracy:", correct*1.0/204)




