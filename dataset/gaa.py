# Game Ad Automation data loader

import re
import os
import keras.utils.image_utils as image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img



class GAADataLoader:
    TRAIN_TXT_PATH="./GAA_DATA/data_set/ImageSets/Main/train.txt"
    VALID_TXT_PATH="./GAA_DATA/data_set/ImageSets/Main/valid.txt"
    IMAGE_PATH ="./GAA_DATA/data_set/JPEGImages/"
    CLOSE_LABEL = 1000

    TYPE_TRAIN = "train"
    TYPE_VALID = "valid"
    
    def __init__(self):
        self.train_db = {}
        self.valid_db = {}
        self.db = {}

        self.db[self.TYPE_TRAIN] = self.train_db
        self.db[self.TYPE_VALID] = self.valid_db

    def add_db(self, kind, label, data):
        if not(label in self.db[kind]):
            self.db[kind][label] = []

        self.db[kind][label].append(data)

    def load_data(self, file_name):
        target_image = image.load_img(self.IMAGE_PATH + file_name)
        #画像をnumpy配列に変換する
        target_image = np.array(target_image).transpose(2,0,1)
        #print("target_image shape=%s" %(str(target_image.shape)))
        return target_image

    def process_ja_char(self, kind, item):
        res = re.match("ja_char_(?P<label>\d+)_(?P<number>\d+)", item)
        if res is None:
            return
    
        label  = int(res.group("label"))
        number = int(res.group("number"))

        data = self.load_data("ja_char_%d_%d.jpg" % (label, number))
        self.add_db(kind, label, data)

    def process_close(self, kind, item):
        res = re.match("closew_(?P<number>\d+)", item)
        if res is None:
            return
    
        number = int(res.group("number"))
        pdb.set_trace()

        data = self.load_data("closew_%d.jpg" % (number))
        self.add_db(kind, self.CLOSE_LABEL, data)


    def load(self):
        path = {self.TYPE_TRAIN: self.TRAIN_TXT_PATH, self.TYPE_VALID: self.VALID_TXT_PATH}
        for kind in [self.TYPE_TRAIN, self.TYPE_VALID]:
            p = path[kind]
            with open(self.TRAIN_TXT_PATH) as f:
                file_list = f.read().splitlines()
        
            for item in file_list:
                print("INFO: processing %s" % (item))
                self.process_ja_char(kind, item)
                self.process_close(kind, item)

import time

#start = time.time()
dl = GAADataLoader()
import pdb
dl.load()
pdb.set_trace()


if __name__ == "main":
    file_name = "./ja_char_0_0.jpg"
    target_image = image.load_img(file_name)
    #画像をnumpy配列に変換する
    target_image = np.array(target_image).transpose(2,0,1)
    print("target_image shape=%s" %(str(target_image.shape)))
    
    print(target_image[0].shape)
    target_image = target_image[0].reshape(64,64,1)
    
    save_img = target_image
    #save_img = array_to_img(target_image, scale = False)
    image.save_img("/tmp/test.jpg", save_img)
    
    
