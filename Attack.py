
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array,array_to_img
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.models import load_model
import preprocess_model as pm
from L-BFGS import box_constrained_attack_with_norm


def prep_data(data_path,n):
    
    with np.load(data_path) as data:
        dataset = data["dataset"]
        labels = data["labels"]
        conf=data['conf']
    
    x=[]
    y=[]
    confidence=[]
    c=0
    preds=model(dataset)
    for i in range(n):
         #print(i)
         label=np.argmax(preds[i])
         conf_=conf[i]
         if (label!=labels[i]):
            c=c+1
            continue 
         x.append(dataset[i])
         confidence.append(conf_) 
         y.append(labels[i])
    
    print(c,"misclassified labels")
    x=np.asarray(x)
    y=np.asarray(y)
    confidence=np.asarray(confidence)
    print("shape",x.shape,y.shape,confidence.shape)
    
    #prep target labels    
    t_labels=np.zeros(y.shape)
    for i in range(n):
        t=1-y[i]
        t_labels[i]=t
        
    #preprocess image
    p_img=np.zeros(x.shape)
    for i in range(n):
        p_img[i]=pm.preprocess_img(x[i])

    #p_img=tf.convert_to_tensor(p_img)
    p_img=tf.cast(p_img,dtype=tf.float32)
    print(p_img.dtype)
    
    return p_img,t_labels

def batch_attack(p_img,t_labels,max_epsilon=10,max_iter=30):
    batch_shape=p_img[:1].shape #shape of the image inputs 
    max_epsilon =max_epsilon  # Max epsilon on the original range (0 to 255)
    max_iter = max_iter # Maximun number of iterations

    eps = 2.0 * max_epsilon / 255.0 # Max epsilon on the range of the processed images (-1 to 1)

    t_lable=t_labels #target label
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy()

    lbfgs_attacker = box_constrained_attack_with_norm(model,batch_shape, max_epsilon=eps, max_iter=max_iter,targeted=True,input=p_img[:1],output=t_labels[:1])   

    attack_imgs=[]
    norms=[]
    confs=[]
    f_labels=[]
    n=len(p_img)
    for i in range(n):
      attack_img,norm,conf,f_label = lbfgs_attacker.generate(p_img[i:i+1],t_labels[i:i+1],verbose=True)
      attack_imgs.append(attack_img)
      if (f_label!=-1):
        norms.append(norm)
        confs.append(conf)
      f_labels.append(f_label)
      
      
    #number of successful adversaries
    c=0
    ind=[]
    for i in range(n):
        if (f_labels[i]!=-1):
            c=c+1
            ind.append(i)
    print("No.of Successful Adversaries:{}",c)
     
    #number of adversaries with more confidence
    d=0
    for i in range(len(confs)):
        print(ind[i])
        if(confs[i]>confidence(ind[i])):
            d=d+1
    print(d)
    
    #computing norm and conf
    conf_avg=np.average(confs)
    conf_max=np.max(confs)
    norm_avg=np.average(norms)
    norm_max=np.max(norms)
    print(conf_avg,conf_max,norm_avg,norm_max)
    
    return conf_avg,norm_avg


def single_img_attack(model,attack,img_path,rand=False):
    """attack:callable attack fn()
        img_path:path of single image to test
        rand: bool, if to test with random input"""
    img = load_img(img_path, color_mode="rgb")
    a_img = img_to_array(img)
    
    if rand:
        #to test with random noise input
        random_img=np.random.randint(50,size=a_img.shape,dtype=np.int32)
    
    p_img = preprocess_img(a_img)
    print(p_img.shape)
    
    #display_images(model,img,"Original")
    l_display_images(model,p_img,"Original",savepath="",save=False,single_img=True)
    print((model(p_img).shape))
    preds=model(p_img)
    label=np.argmax(preds)
    print(label.shape)
    
    #calling attack
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy()
    eps=0.7
    t_lables=np.array([[1-label]])
    adv_x,noise,norm,conf,f_label=attack(model,p_img,eps,np.inf,clip_min=-1,clip_max=1,y=t_lables,targeted=True,loss_fn=loss_fn)
    l_display_images(model,adv_x,'eps-1')
    l_display_images(model,noise,'noise')
    print(norm)
    

if __name__=="main":
    path=input("Model path")
    
    model=load_model(path)
    
    data_path=input("Test data path")
    n=input("Number of images")
    
    p_img,t_labels=prep_data(data_path,n)
    eps=input("Max epsilon")
    iters=input("Max iters")
    
    conf_avg,norm_avg=batch_attack(p_img, t_labels,eps,iters)