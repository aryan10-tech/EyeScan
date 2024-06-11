import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to suppress unnecessary messages

import tensorflow as tf
print(tf.__version__) #2.16.1
from tensorflow.keras.optimizers import RMSprop  # Import RMSprop optimizer
import numpy as np
import cv2
from PIL import Image

labels = {0: 'Cataract', 1: 'Diabetic Retinopathy', 2: 'Dry Amd', 3: 'Glaucoma', 
          4: 'Hypertensive Retinopathy', 5: 'Lupus Retinopathy', 6: 'Normal', 7: 'Retinal Hemorrhage', 
          8: 'Sickle Cell Retinopathy', 9: 'Wet Amd'}


cnn = tf.keras.models.load_model('Ocular_main.h5', custom_objects={'RMSprop': RMSprop})

def processed_img(image_path):
    
    imag = cv2.imread(image_path)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    # Predict using the loaded model
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions[0])  # Find the index of the maximum value
    print(result_index)
    predicted_class = labels[result_index]
    print(f"It's a {predicted_class}")

    return predicted_class


