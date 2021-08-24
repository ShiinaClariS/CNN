# -*- coding: utf-8 -*-
"""
author: ShiinaClariS
time: 2021年8月24日13:37:48
"""
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class ReadImage:
    dog_img_dir = []
    X = []
    y_label = []
    x_train, y_train, x_test, y_test = None, None, None, None

    def __init__(self, kind):
        self.read(kind)

        for i in range(kind):
            self.training_data(label=self.dog_img_dir[i].split('-', 1)[1], data_dir=self.dog_img_dir[i])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.y_label)
        y = to_categorical(y, kind)

        x = np.array(self.X)
        x = x / 255

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    def read(self, kind):
        root_dir = r"W:\\Dog_CNN\\images\\Images\\"
        # root_dir = path

        dogs_img_list = os.listdir(r'W:\\Dog_CNN\\images\\Images')
        # dogs_img_list = os.listdir(path)
        for i in dogs_img_list:
            print(i)

        for i in range(kind):
            self.dog_img_dir.append(root_dir + dogs_img_list[i])

    def training_data(self, label, data_dir):
        img_size = 150

        for img in os.listdir(data_dir):
            # print(img)
            path = os.path.join(data_dir, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size))
            self.X.append(img)
            self.y_label.append(str(label))
