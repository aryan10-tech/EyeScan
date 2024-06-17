import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress unnecessary TensorFlow logging messages

import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop
import numpy as np 
import cv2  

# Dictionary to map prediction indices to their corresponding medical conditions
labels = {0: 'Cataract', 1: 'Diabetic Retinopathy', 2: 'Dry Amd', 3: 'Glaucoma', 
          4: 'Hypertensive Retinopathy', 5: 'Lupus Retinopathy', 6: 'Normal', 7: 'Retinal Hemorrhage', 
          8: 'Sickle Cell Retinopathy', 9: 'Wet Amd'}

# Load the pre-trained CNN model from a file
cnn = tf.keras.models.load_model('Ocular_main.h5', custom_objects={'RMSprop': RMSprop})

def processed_img(image_path):
    imag = cv2.imread(image_path)  # Read the image using OpenCV
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))  # Load and resize the image
    input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Convert the image to a numpy array
    input_arr = np.array([input_arr])  # Expand dimensions to create a batch
    
    predictions = cnn.predict(input_arr)  # Predict the class probabilities
    result_index = np.argmax(predictions[0])  # Get the index of the highest probability class
    
    predicted_class = labels[result_index]  # Map the index to the corresponding label
    print(f"It's a {predicted_class}")  # Print the predicted class

    return predicted_class  # Return the predicted class
