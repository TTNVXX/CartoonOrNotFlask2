from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
