# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:51:31 2021

@author: Leo
"""
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import tensorflow as tf
# SC=[]
# AUD1=tf.keras.Model()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=7, min_lr=0.000001)


class FileManager():
    def __init__(self, N, dv, k):
        self.N = N
        self.dv = dv
        self.k = k

    def DataSaver(self,SC,AUD1):
        model_name = 'dv' + str(self.dv) + '_k' + str(self.k) + '.h5'
        sc_name = 'dv' + str(self.dv) + '_k' + str(self.k) + '.csv'
        dfsc = pd.DataFrame(SC)
        dfsc.to_csv(sc_name, index=False)
        AUD1.save_weights(model_name)

    def ModelLoader(self,AUD1):
        name = 'dv' + str(self.dv) + '_k' + str(self.k) + '.h5'
        AUD1.load_weights(name)

    def SCLoader(self):
        name = 'dv' + str(self.dv) + '_k' + str(self.k) + '.csv'
        sc = pd.read_csv(name)
        sc = sc.to_numpy()
        sc_list = []
        for i in range(self.N):
            sc_list = sc_list + [sc[i, :].tolist()]
        return sc_list
