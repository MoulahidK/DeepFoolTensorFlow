import deepfool_tf
from deepfool_tf import deepfool
import os 

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize

from Laye import *


# from Layer import *

def test_deepfool(model_file='', pic_path=''):
        Label = { 0: "Cat", 1: "Dog", 2: "Human"}
        image = Image.open(pic_path)
        print(image.size)
        image = image.resize((224,224))
        # print(image.size)
        # model = tf.keras.models.load_model('Conv2D_3_Layers.h5')
        model = tf.keras.models.load_model('Gaussian_3_Layers.h5', custom_objects = {"FTGDConvLayerInference": FTGDConvLayerInference})
        model.summary()
        # print("heer")

        r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model)
        print("label_orig: ", Label[label_orig])
        print("label_pert: ", Label[label_pert])
        print(pert_image.shape)
        pert_image = np.reshape(pert_image, (224, 224, 3))
        print()
        pert_image += 0.5
        pert_image *= 255
        
        ### Generating attacked images Train ##########
        # final_image = Image.fromarray(pert_image.astype(np.uint8))
        # basename = os.path.basename(filename.path)
        # subdirname = os.path.basename(os.path.dirname(filename.path))
        # print(basename)
        # print(subdirname)
        # image_path = "Hacked_Train_Conv/"+subdirname+"/"
        # final_image.save(f"{image_path}Hacked_"+basename,"JPEG")
        
         #### Generating attacked images Test##########
        # final_image = Image.fromarray(pert_image.astype(np.uint8))
        # basename = os.path.basename(filename.path)
        # subdirname = os.path.basename(os.path.dirname(filename.path))
        # print(basename)
        # print(subdirname)
        # image_path = "../Hacked_Test_Conv/"+subdirname+"/"
        # final_image.save(f"{image_path}Hacked_"+basename,"JPEG")
      
         ### Generating attacked images Train ##########
        # final_image = Image.fromarray(pert_image.astype(np.uint8))
        # basename = os.path.basename(filename.path)
        # subdirname = os.path.basename(os.path.dirname(filename.path))
        # print(basename)
        # print(subdirname)
        # image_path = "Hacked_Train_Gauss/"+subdirname+"/"
        # final_image.save(f"{image_path}Hacked_"+basename,"JPEG")
        
      



        # look at the original image and the adversarial sample with the predited classes

        # plt.figure()
        # plt.subplot(1, 2, 1), plt.imshow(image[0,:,:,:])
        # plt.subplot(1, 2, 1), plt.title('Original Image')
        # plt.subplot(1, 2, 2), plt.imshow(pert_image[0,:,:,:])
        # plt.subplot(1, 2, 2), plt.title('Adversarial Sample')

        # print('\nRESULTS :')

        # print('Original Image : ')
        # print('Adversarial Sample :')

        plt.figure()
        plt.imshow(pert_image[0,:,:])
        plt.title(Label[label_pert])
        plt.show()
        plt.show()
                
        
directories = "Train"
for directory in os.scandir(directories):
    # print(directory)
    # for filename in os.scandir(directory): 
    #     # print(filename.path)
    #     test_deepfool('',filename.path)
       
    file = random.choice(os.listdir(directory)) 
    path = directory.path+"/"+file
    print(path)
    test_deepfool('',path)

# adversarial_robustness = (1 / len) * ((torch.norm(rt.flatten()) / (torch.norm(x.flatten()))

