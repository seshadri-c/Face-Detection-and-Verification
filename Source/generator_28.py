from keras.utils import Sequence
import numpy as np
import glob
import pandas as pd
import cv2

import os
import matplotlib.pyplot as plt

DATA_PATH="/home/ecsuiplab/faceverification/yolo3/src_veri/olivetti faces/"
#DATA_PATH="/home/dhriti/Documents/ISI/project/src_veri/olivettifaces/"
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class ImageSequence(Sequence):
    def __init__(self, batch_size=1,input_size=(256,256)):
        self.image_seq_path=DATA_PATH
        self.input_shape=input_size
        self.epoch = 0
	self.batch_size=1
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.dirs=os.listdir(self.image_seq_path)
        self.num_dir=len(self.dirs)


    def read_image(self,file_path):
        Img=cv2.imread(file_path)
        if(len(Img.shape)==3):
            Img = rgb2gray(Img)
            Img=cv2.resize(Img,(256,256))
            Img=Img[:,:,np.newaxis]/255.0

        return Img

    def __getitem__(self, idx):

        id1=np.random.randint(0,self.num_dir) 
        dir1=self.image_seq_path+self.dirs[id1]+"/"
	
        files1=os.listdir(dir1)
        file_path1=dir1+files1[np.random.randint(len(files1))]

        I1=self.read_image(file_path1)
       
        return (I1,file_path1)

    

