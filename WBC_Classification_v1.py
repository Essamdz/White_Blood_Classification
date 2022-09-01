import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from keras.models import load_model

class wbcType:
    def __init__(self):
        model_url = ".\imagenet_efficientnet_v2_imagenet21k_s_feature_vector_2"
        self.module = hub.KerasLayer(model_url)
        self.model= load_model('my_model.h5')
        
    def deepfeature(self,image):
        image = np.array(image)
        if len(image.shape)==2:
            backtorgb = cv2.cvtColor(np.array(image, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
            image=backtorgb
        image=cv2.resize(image,(384,384))
        # reshape into shape [batch_size, height, width, num_channels]
        img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
        logits=self.module(image)
        return np.array(logits)[0]
    
    def predictType(self,path):
        im=cv2.imread(path)[:,:,::-1]
        deepf=self.deepfeature(im)
        target_names=["Lymph","Neut","BASO","EOSI","MONO"]
        pred=self.model.predict([deepf.tolist()])
        return target_names[pred.argmax(axis=1)[0]]

    
wbc=wbcType()  



#############   Example 1 #######################
path="file_290_6236.png"
im=cv2.imread(path)[:,:,::-1]
plt.imshow(im)
plt.show()
res=wbc.predictType(path)  
print(res)
#############   Example 2  #######################
path="file_44_569.png"
im=cv2.imread(path)[:,:,::-1]
plt.imshow(im)
plt.show()
res=wbc.predictType(path)  
print(res)


#############   Example 3  #######################
path="file_242_6044.png"
im=cv2.imread(path)[:,:,::-1]
plt.imshow(im)
plt.show()
res=wbc.predictType(path)  
print(res)
