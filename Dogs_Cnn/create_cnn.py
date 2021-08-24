# -*- coding: utf-8 -*-
"""
author: ShiinaClariS
time: 2021年8月24日13:38:34
"""
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class CreateCnn:
    def __init__(self, kind):
        self.cnn = models.Sequential()

        self.cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        self.cnn.add(MaxPooling2D((2, 2), input_shape=(148, 148, 32)))

        self.cnn.add(Conv2D(64, (3, 3), activation='relu', input_shape=(74, 74, 32)))
        self.cnn.add(MaxPooling2D((2, 2), input_shape=(72, 72, 64)))

        self.cnn.add(Conv2D(128, (3, 3), activation='relu', input_shape=(36, 36, 64)))
        self.cnn.add(MaxPooling2D((2, 2), input_shape=(34, 34, 128)))

        self.cnn.add(Conv2D(128, (3, 3), activation='relu', input_shape=(17, 17, 128)))
        self.cnn.add(MaxPooling2D((2, 2), input_shape=(15, 15, 128)))

        self.cnn.add(Flatten(input_shape=(7, 7, 128)))

        self.cnn.add(Dense(512, activation='relu'))

        self.cnn.add(Dense(kind, activation='softmax'))

        self.cnn.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['acc'])
