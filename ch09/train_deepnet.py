# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer
from dataset.gaa import * 

print("now dir")
print(os.getcwd())
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
(x_train, t_train), (x_test, t_test) = GAADataLoader().load_gaa_data()

#os.environ["OPENBLAS_NUM_THREADS"] = "64"

network = DeepConvNet(input_dim=(3,64,64),output_size=1001) #ja_chars and closew 
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=200, mini_batch_size=1000,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
#import pdb
#pdb.set_trace()

trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
