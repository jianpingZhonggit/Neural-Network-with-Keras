# coding=utf-8
import os
# 定义类别字典
class_dict = {"animal": 1, "flower": 2, "guitar": 3, "houses": 4, "plane": 5}
folder_list = os.listdir('./dataset/val/')
f = open('./test.txt', 'w')
for folder in folder_list:
    img_list = os.listdir("./dataset/val/"+folder)
    for img in img_list:
        f.write("val/"+folder+"/"+img+" "+str(class_dict[folder])+"\n")


