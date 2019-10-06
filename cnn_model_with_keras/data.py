import os
import cv2
import numpy as np
img_label_dict = {"animal": 0, "flower": 1, "guitar": 2, "houses": 3, "plane": 4}


class ImageGenerator:
    def __init__(self, folder_path, img_width, img_height, batch, number_class):
        self.folder_list = os.listdir(folder_path)
        self.img_width = img_width
        self.img_height = img_height
        self.index = 0
        self.number_class = number_class
        self.batch = batch
        self.img_list, self.label_list = self.get_list(self.folder_list)

    @staticmethod
    def get_list(folder_list):
        img_list = []
        label_list = []
        for folder in folder_list:
            for _ in os.listdir('../dataset/train/'+folder):
                img_list.append(folder+'/'+_)
            label_list = [img_label_dict[folder]] * len(os.listdir('../dataset/train/'+folder))
        return img_list, label_list

    def get_batch(self):
        print(len(self.img_list), "*" * 30)
        while True:
            images = []
            labels = []
            for i in range(self.batch):
                try:
                    img = cv2.imread("../dataset/train/"+self.img_list[i+self.index])
                except:
                    print(i+self.index)
                img = cv2.resize(img, (self.img_width, self.img_height))
                images.append(img)
                label = [0] * (self.number_class + 1)
                label[self.label_list[i]] = 1
                labels.append(label)
            images = np.array(images)
            labels = np.array(labels)
            yield images, labels
            if self.index > len(self.img_list) - 2*self.batch:
                self.index = 0
            else:
                self.index += self.batch




