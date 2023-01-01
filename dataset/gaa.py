# Game Ad Automation data loader

import re
import os
import pickle
import keras.utils.image_utils as image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img


class GAADataLoader:
    TRAIN_TXT_PATH="./GAA_DATA/data_set/ImageSets/Main/train.txt"
    VALID_TXT_PATH="./GAA_DATA/data_set/ImageSets/Main/val.txt"
    IMAGE_PATH ="./GAA_DATA/data_set/JPEGImages/"
    SAVE_PATH ="./GAA_DATA/gaa_data.pkl"
    CLOSE_LABEL = 1000

    TYPE_TRAIN = "train"
    TYPE_VALID = "valid"
    
    def __init__(self):
        self.train_db = {}
        self.valid_db = {}
        self.db = {}

        self.db[self.TYPE_TRAIN] = self.train_db
        self.db[self.TYPE_VALID] = self.valid_db

        self.x_train = []
        self.t_train = []
        self.x_test  = []
        self.t_test  = []
        
        self.data_set = []

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

        data = self.load_data("closew_%d.jpg" % (number))
        self.add_db(kind, self.CLOSE_LABEL, data)


    def init_gaa_data(self):
        path = {self.TYPE_TRAIN: self.TRAIN_TXT_PATH, self.TYPE_VALID: self.VALID_TXT_PATH}
        for kind in [self.TYPE_TRAIN, self.TYPE_VALID]:
            with open(path[kind]) as f:
                file_list = f.read().splitlines()
      
            num = 0
            total = len(file_list)
            report_num = int(total / 20.0)
            print("INFO: %s , report_num = %d, total=%d)" % (kind, report_num, total))
            for item in file_list:
                self.process_ja_char(kind, item)
                self.process_close(kind, item)
                num += 1
                if num % report_num == 0:
                    print("INFO: %s processing %d percent" % (kind, int((num/total*100.0))))


        #for train
        kind = self.TYPE_TRAIN
        for label, data_list in self.db[kind].items():
            for data in data_list:
                self.x_train.append(data)
                self.t_train.append(label)

        #for test
        kind = self.TYPE_VALID
        for label, data_list in self.db[kind].items():
            for data in data_list:
                self.x_test.append(data)
                self.t_test.append(label)

        with open(self.SAVE_PATH, "wb") as f:
            self.data_set = [self.x_train, self.t_train, self.x_test, self.t_test]
            pickle.dump(self.data_set,f,-1)


    def load_gaa_data(self):
        if not os.path.exists(self.SAVE_PATH):
            self.init_gaa_data()

        with open(self.SAVE_PATH, 'rb') as f:
            self.data_set = pickle.load(f)

        self.x_train = self.data_set[0]
        self.t_train = self.data_set[1]
        self.x_test  = self.data_set[2]
        self.t_test  = self.data_set[3]

        return (self.x_train, self.t_train), (self.x_test, self.t_test)


def some_test():
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

def gaa_data_init_main():
    print("INFO: init gaa data")
    import time
    
    start = time.time()
    dl = GAADataLoader()
    dl.init_gaa_data()
    end = time.time()
    
    print("INFO: elapsed time = %d sec" % (end-start))
#    import pdb
#    pdb.set_trace()

def gaa_data_load_main():
    print("INFO: load gaa data")
    import time
    
    start = time.time()
    dl = GAADataLoader()
    dl.load_gaa_data()
    end = time.time()
    
    print("INFO: elapsed time = %d sec" % (end-start))
#    import pdb
#    pdb.set_trace()



if __name__ == "__main__":
    #gaa_data_init_main()
    gaa_data_load_main()
