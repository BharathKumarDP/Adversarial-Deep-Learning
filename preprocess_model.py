
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array,array_to_img,save_img
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


#preprocess input image for ResNet50(caffe) 
def preprocess_img(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)
    image = image[None, ...]
    
    return image

#for reverse pre-process from array
def restore_original_image_from_array(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x

def reverse_preprocess(p_img,a_numpy=True):
    if a_numpy:
      p_img=p_img.numpy()
    ip_img=restore_original_image_from_array(p_img, data_format='channels_last')
    ia_img=array_to_img(ip_img[0])
    img=ia_img
    return img


#helper function to display images
def l_display_images(model,image,description,savepath=None,save=False,single_img=True,categorical=True):

    classes={0:'Unaffected',1:'DR Affected'}
    i=0
    #if more than one image
    if (single_img==False):
      for img in image:
        preds=model.predict(img)
        if categorical:
            label=np.argmax(preds)
            class_confidence=preds[i][label]
        else:
            label = (preds > 0.5).reshape(1).astype(np.int)[0]
            class_confidence=preds.reshape(1).astype(np.float32)[0]
        
        print(label)
        image_class=classes[label]
      
        print("result",label,image_class, class_confidence)

        img=reverse_preprocess(img)
        plt.figure()
        plt.imshow(img)
        plt.title('{} \n {} : {:.2f}%'.format(description,
                                                        image_class,class_confidence*100))
        plt.show()
        if save:
          path=savepath+'img{}'.format(i)
          save_img(path)
        i=i+1
        print('completed')
    
    #single image
    else:

      preds=model.predict(image)
      if categorical:
          label=np.argmax(preds)
          class_confidence=preds[0][label]
      else:
          label = (preds > 0.5).reshape(1).astype(np.int)[0]
          class_confidence=preds.reshape(1).astype(np.float32)[0]
        
      print(label)
      image_class=classes[label]
      print("result",preds,image_class, class_confidence)

      img=reverse_preprocess(image)
      plt.figure()
      plt.imshow(img)
      plt.title('{} \n {} : {:.2f}%'.format(description,
                                                      image_class,class_confidence*100))
      plt.show()
      if save:
        path=savepath+'img{}'.format(i)
        save_img(path)
   
#function to shuffle numpy arrays a,b,c in unison     
def shuffle_in_unison_triv(a,b,c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)

def l2(x, y):
  # technically squarred l2
    return tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape))))

