import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to suppress unnecessary messages

from PIL import Image  # Import the Python Imaging Library (PIL) for image processing
from keras.preprocessing.image import load_img, img_to_array  # Import functions for loading and converting images
import numpy as np  # Import numpy for numerical operations
from keras.models import load_model  # Import function for loading a trained model

# Load the pre-trained model
model = load_model('Ocular_main.h5')

# Define a dictionary mapping class indices to class labels
labels = {0: 'Cataract', 1: 'Diabetic Retinopathy', 2: 'Dry Amd', 3: 'Glaucoma', 
          4: 'Hypertensive Retinopathy', 5: 'Lupus Retinopathy', 6: 'Normal', 7: 'Retinal Hemorrhage', 
          8: 'Sickle Cell Retinopathy', 9: 'Wet Amd'}

# Define a function to process an input image and predict the class
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))  # Load and resize the image to match the input size of the model (224x224 pixels)
    img = img_to_array(img)  # Convert the image to a numpy array
    img = img / 255  # Normalize the pixel values to the range [0, 1]
    img = np.expand_dims(img, [0])  # Add an extra dimension to the array to represent batch size (1 in this case)
    answer = model.predict(img)  # Use the trained model to predict the class of the input image
    y_class = answer.argmax(axis=-1) # Get the index of the predicted class with the highest probability
    result_ocular = labels[y_class[0]] # Map the index to the corresponding class label using the labels dictionary
    # Return the predicted class label
    return result_ocular

img_path='C:/Users/HP/Desktop/Dream_Anjali/EasyCheck/Fundus Endoscopy/cataract/_100_334408.jpg'
predicted_result = processed_img(img_path)
print(predicted_result)
