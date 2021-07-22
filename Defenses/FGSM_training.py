
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array,array_to_img,save_img
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess_model import preprocess_img
from FGSM_attack import fast_gradient_method

"""This file is to create Adversarial Datasets.
    Adversarial Training i.e Training with perturbed dataset is a defense method and so we prepared
    adversaries with FGSM attack() and trained the model with the original labels.
    
    This training didn't prevent the attack but made the model bit more robust and harder to attack.More results in Documentation"""

loss_fn= tf.keras.losses.SparseCategoricalCrossentropy()
#calls FGSM attack on each image and stores it with the origianl target label
def FSGM_training(model,eps,dataset,lables,to_save=False,savepath=''):
       """model: Callable (function) that accepts an input tensor 
                    and return the model logits (unormalized log probs)
      eps: epsilon value for FGSM attack
      dataset: input tensor images
      lables: target lables
      to_save:if to save the adversary dataset"""
      adv_imgs=[]
      i=0
      a=0
      b=0
      for img in dataset:
        #img = tf.cast(img, tf.float32)
        #img = preprocess_input(img)
        img=preprocess_img(img)
        lable=lables[i]
        if lable:
          b=b+1
        else:
          a=a+1
        t_lable=np.abs(1-lable).reshape(1,1)
        adv_x=fast_gradient_method(model,img,eps,np.inf,clip_min=-1,clip_max=1,y=t_lable,targeted=True,loss_fn=loss_fn)
        print(i)
        print('a,b',a,b)
        preds = (model.predict(adv_x) > 0.5).astype(np.int)
        if (t_lable!=preds):
          print("error....",i)
        adv_x=np.squeeze(adv_x)
        adv_imgs.append(adv_x)
        i=i+1
      
      print(len(adv_imgs))
      dataset=adv_imgs
      if to_save:
          np.savez(savepath, dataset=adv_imgs,labels=labels)
      return adv_imgs

