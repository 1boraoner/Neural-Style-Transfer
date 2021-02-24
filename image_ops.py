import tensorflow as tf
from tensorflow.keras.preprocessing import image as img
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np

def img_read(name):
    img_arr = img.load_img('C:\\Users\\user\\Desktop\\NST\\images\\' + str(name), target_size=(224,224))
    img_arr = img.img_to_array(img_arr)
    return img_arr

def img_show(img_arr):
    plt.figure(figsize=(8,8))
    plt.imshow(img_arr)
    plt.show()


def random_image(content,noise_ratio = 0.6):
    size = content.shape
    noise = np.random.randn(size[0],size[1],size[2])
    generated = noise*noise_ratio + (1-noise_ratio)*content
    return generated

def vgg_input(image):   # make vgg input format (1,nh,nw,nc)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)
