
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import tensorflow as tf

"""This attack is originally proposed by Schezdy et al in their paper about Adversarial Attacks.
Paper:
Source:
This is a iterative attack that finds adversarial examples by optimizing a custom function using
L-BFGS() optimization method. The authors used a box-constrained variant of the method.
This method produces a highly confident adversaries but is computationally slow.
A modification of this attack without norm constraint was trained and it didn't give better results and
so we stuck with this method"""

class box_constrained_attack_with_norm:
    """ Creates adversarial samples using box contrained L-BFGS
    """
    def __init__(self,
                 model,
                 batch_shape,
                 max_epsilon,
                 max_iter,
                 targeted,
                 input,
                 output=None,
                 loss_fn=None,
                 img_bounds=(-1, 1),
                 use_noise=True,
                 max_ls=5,
                 n_classes=1001,
                 rng=np.random.RandomState()):
        """ 
             model: Callable (function) that accepts an input tensor 
                    and return the model logits (unormalized log probs)
             batch_shape: Input shapes (tuple). 
                    Usually: [batch_size, height, width, channels]
             max_epsilon: Maximum L_inf norm for the adversarial example
             (maximum perturbation allowed for any pixel)
             max_iter: Maximum number of iterations (gradient computations)
             (higher value gives confident results,leads to more distortion)
             targeted: Boolean: true for targeted attacks, false for non-targeted attacks
             img_bounds: Tuple [min, max]: bounds of the image. Example: [0, 255] for
                    a non-normalized image, [-1, 1] for inception models.
             max_ls: Maximum number of line searches
             (for one dimensional binary search of c)
        """
        self.x_input =input

        if output is None:
          self.y_input = np.argmax(model(input)).reshape(batch_shape[0],1) #untargeted, uses the predicted lable 
        else:
          self.y_input=output #targeted, target lable
      
        if loss_fn is None:
           self.loss_fn= tf.keras.losses.SparseCategoricalCrossentropy()
        else:
           self.loss_fn=loss_fn

        #computing initial gradient of loss wrt to inputs
        with tf.GradientTape() as gt:
          gt.watch(self.x_input) 
          lables = model(self.x_input)
          self.loss=self.loss_fn(self.y_input,lables)
          
        self.grad = gt.gradient(self.loss, self.x_input)

        self.targeted = targeted
        self.max_iter = max_iter
        self.max_epsilon = max_epsilon
        self.batch_shape = batch_shape
        self.img_bounds = img_bounds
        self.use_noise = use_noise
        self.rng = rng
        self.max_ls = max_ls
        self.adversary=input
        self.model=model
    
    # technically squarred l2(source:cleverhans)
    def l2(self,x, y):
        return tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape))))
           
    #function to generate adversaries
    def generate(self,images,t_labels,verbose=False):
        """ Generates adversarial images/
            images: a 4D tensor containing the original images
            t_labels: for non-targeted attacks, the actual or predicted labels
                               for targeted attacks, the desired target classes for each image.         
            returns: adv_images: a 4D tensor containing adversarial images
        """
        
        #lower and upper bounds to clip the image(eps value is used here)
        lower_bounds = np.minimum(-1 - images, -self.max_epsilon).reshape(-1)
        upper_bounds = np.maximum(1 - images, self.max_epsilon).reshape(-1)
        bounds = list(zip(lower_bounds, upper_bounds))
     
       #to use random starting point
        if self.use_noise:
            alpha = self.max_epsilon * 0.5
            x0 = alpha * np.sign(np.random.random(np.prod(images.shape)))
        else:
            x0 = np.zeros(np.prod(images.shape))

        #targeted case, use given target lables
        if self.targeted:
              self.t_label=t_labels
        else:
              self.t_label=np.argmax(self.model(self.x_input)).reshape(self.batch_shape[0],1)
        
        #batch of input images to consider together
        self.input=images
    
        """function to optimize using l-bfgs
        #inputs:
            func- objective function to optimize
            c-constant term in function(refer paper)
        #outputs(bool)- the adverarial image and if the image is successful"""
        def lbfgs(func,c):
          delta_best, f, d = fmin_l_bfgs_b(func=func,
                                              x0=x0,
                                              args=(c, ),
                                              bounds=bounds,
                                              maxfun=self.max_iter,
                                              maxls=self.max_ls)
          return self.is_adversary(images + delta_best.reshape(images.shape).astype(np.float64),self.t_label),f,d


        #to search for constant c
        # finding initial c,eps is used here
        c = self.max_epsilon
        print('intial c',c)
        x0 = images
        #range of c values are used and lbfgs is called with each c value, to find the smallest c value which gives an adv
        for i in range(5):
            c = 2 * c
            print('c={}'.format(c))
            is_adversary,_,_ = lbfgs(self.func, c)
            if is_adversary==True:
                print('adv',is_adversary,c)
                break
        if not is_adversary:#if adversary isnt found in this range of c values, return failed.
            print('failed')
            return self.adversary,0,0,-1 #TODO: these outputs should be handled in return,-1 lable means Failed

        # binary search c
        print('binary search c...')
        c_low = 0
        c_high = c
        #binary search done on c, till the range becomes less than epsilon
        #high epsilon,a large c value is found
        while c_high - c_low >= self.max_epsilon:
            print('c_high={}, c_low={}, diff={}, epsilon={}'
                         .format(c_high, c_low, c_high - c_low,self.max_epsilon))
            c_half = (c_low + c_high) / 2
            is_adversary,_,_ = lbfgs(self.func, c_half)
            if is_adversary:
                c_high = c_half
            else:
                c_low = c_half

        #computing final params to return
        adv_img=self.adversary
        norm=self.l2(adv_img,images)
        
        preds=self.model.predict(adv_img)
        lables=np.argmax(preds,axis=1)
        print(preds,lables)
        
        confidence=np.max(preds,axis=1)*100
        print(confidence)

        return adv_img,norm,confidence,lables


    #objective function,retuns loss and gradient
    def func(self,delta,c):
        images=self.input
        attack_img = images+ delta.reshape(images.shape).astype(np.float64)
        
        print(self.model(attack_img))
        loss_fn=self.loss_fn
        
        with tf.GradientTape() as gt:
          gt.watch(attack_img)
          loss=loss_fn(self.t_label,self.model(attack_img))

        grad=gt.gradient(loss,attack_img)
        grad=tf.reshape(grad,-1)

        #sqaured l2 distance * c (refer paper)    
        norm=c*self.l2(attack_img,images)
        #print("norm",norm)

        loss= loss.numpy().astype(np.float64)
        grad=grad.numpy().astype(np.float64)
        
        #final value to optimize
        loss=loss+norm
        #print("loss",loss)
        if self.targeted:
            # Multiply by -1 since we want to maximize it.
            return  loss,grad
        else:
            return -1*loss,-1*grad
   
    #helper function to check if adversary has been created
    def is_adversary(self,adv_img,t_label):
      p_label=np.argmax(self.model(adv_img))
      print('is_adv',p_label)
      if self.targeted:
        if t_label==p_label:
          self.adversary=adv_img
          return True
        else:
          return False
      else:
        if t_label!=p_label:
          self.adversary=adv_img
          return True
        else:
          return False


          
      

