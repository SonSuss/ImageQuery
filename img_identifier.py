

# import os

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import keras
# from keras.models import load_model
# import numpy as np
# from PIL import Image


# model = load_model('traffic_sign_identifier.h5')

# model.summary()

# # Function to preprocess input image
# def preprocess_image(image_path):
#     # Load the image
#     img = Image.open(image_path)
#     # Resize the image to match the input shape expected by the model
#     img = img.resize((32, 32))
#     # Convert the image to a numpy array
#     img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
#     # Expand the dimensions to create a batch of size 1
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Function to make predictions
# def predict_image(image_path):
#     # Preprocess the input image
#     preprocessed_image = preprocess_image(image_path)
#     # Use the model to make predictions
#     predictions = model.predict(preprocessed_image)
#     # Get the predicted class
#     predicted_class = np.argmax(predictions[0])
#     return predicted_class

# # Path to the PNG input image
# image_path = './test03.jpg'

# # Make predictions on the input image
# predicted_class = predict_image(image_path)

# # Print the predicted class
# print("Predicted class:", predicted_class)

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = load_model('traffic_sign_identifier.h5')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize image to match model input shape
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values (assuming image is in RGB format)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load and preprocess the image
image_path = './test01.png'
img = Image.open(image_path)
img = img.convert('RGB')  # Convert RGBA to RGB
img = img.resize((32, 32))  # Resize image to match model input shape
img = np.array(img)  # Convert image to numpy array
img = img / 255.0  # Normalize pixel values
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(img)

# Get the predicted class label
predicted_class = np.argmax(predictions)

print("Predicted class:", predicted_class)

