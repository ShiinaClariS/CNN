# -*- coding: utf-8 -*-
"""
author: ShiinaClariS
time: 2021年8月24日13:39:10
"""
from Dogs_Cnn.read_image import ReadImage
from Dogs_Cnn.create_cnn import CreateCnn


class Run:
    def __init__(self, kind):
        r = ReadImage(kind)
        x_train, y_train, x_test, y_test = r.x_train, r.y_train, r.x_test, r.y_test

        c = CreateCnn(kind)
        cnn = c.cnn

        self.history = cnn.fit(x=x_train,
                               y=y_train,
                               epochs=50,
                               batch_size=256,
                               validation_data=(x_test, y_test))

    def get(self):
        print(self.history.history)
        print(self.history.epoch)

        return self.history.history, self.history.epoch


if __name__ == '__main__':
    run = Run(10)
    run.get()
