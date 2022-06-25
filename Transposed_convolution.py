import tensorflow as tf 
from tensorflow import keras 
from keras.models import Model 
from keras import Input 
from keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.utils import plot_model 
import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.model_selection import train_test_split 
import cv2 
import matplotlib 
import matplotlib.pyplot as plt 
import graphviz 
import sys
import os

main_dir=os.path.dirname(sys.path[0])
print(main_dir)

LocImg=main_dir+"data/101_ObjectCategories/"

CATS = set(["dolphin", "butterfly", "cougar_face", "elephant"])
#CATS = set(["panda"])

Imgpath=[]
for cat in CATS:
    for img in list(os.listdir(LocImg+cat)):
        Imgpath=Imgpath+[LocImg+cat+"/"+img]
        
lowres_data=[]
hires_data=[]
for img in Imgpath:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lowres_img = cv2.resize(image, (64, 64))
    hires_img = cv2.resize(image, (256, 256))
    lowres_data.append(lowres_img)
    hires_data.append(hires_img)
    
lowres_data = np.array(lowres_data, dtype="float") / 255.0
hires_data = np.array(hires_data, dtype="float") / 255.0

trX, tsX, trY, tsY = train_test_split(lowres_data, hires_data, test_size=0.2, random_state=0)

fig, axs = plt.subplots(2, 3, sharey=False, tight_layout=True, figsize=(16,9), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,3):
        axs[i,j].matshow(trX[n])
        n=n+1
plt.show() 

fig, axs = plt.subplots(2, 3, sharey=False, tight_layout=True, figsize=(16,9), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,3):
        axs[i,j].matshow(trY[n])
        n=n+1
plt.show() 

input_shape=(trX.shape[1],trX.shape[2],trX.shape[3]) 

inputs = Input(shape=input_shape, name='Input-Layer')

d = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same", name='Transpose-Conv2D-Layer-1')(inputs)
d = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same", name='Transpose-Conv2D-Layer-2')(d)
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation="sigmoid", padding="same", name='Output-Layer')(d)

mdl = Model(inputs, outputs, name='Transposed-Convolutional-NN')
mdl.compile(optimizer="adam", loss="mse", metrics=["Accuracy"])

mdl.summary()

plot_model(mdl, show_shapes=True, dpi=300)

history = mdl.fit(trX, trY, epochs=200, batch_size=4, verbose=1, validation_data=(tsX, tsY), validation_freq=10)

train_img_indx=6
test_img_indx=5
train_image=trX[train_img_indx]
test_image=tsX[test_img_indx]

train_image = train_image[np.newaxis, ...]
test_image = test_image[np.newaxis, ...]

train_img_upscale = mdl.predict(train_image)
test_img_upscale = mdl.predict(test_image)

train_img_upscale=train_img_upscale.reshape(256, 256, 3)
test_img_upscale=test_img_upscale.reshape(256, 256, 3)

fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True, figsize=(16,9), facecolor='white')
axs[0].matshow(trX[train_img_indx])
axs[0].set(title='Original Low-Res')
axs[1].matshow(train_img_upscale)
axs[1].set(title='Reconstructed Hi-Res')
axs[2].matshow(trY[train_img_indx])
axs[2].set(title='Original Hi-Res')
plt.show() 

fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True, figsize=(16,9), facecolor='white')
axs[0].matshow(tsX[test_img_indx])
axs[0].set(title='Original Low-Res')
axs[1].matshow(test_img_upscale)
axs[1].set(title='Reconstructed Hi-Res')
axs[2].matshow(tsY[test_img_indx])
axs[2].set(title='Original Hi-Res')
plt.show()  