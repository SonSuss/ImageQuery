import numpy as np
from keras.preprocessing import image
from keras.models import load_model

def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.  # Normalize pixel values
    return img

def recognize_image(img_path, model_path):
    model = load_model(model_path)
    img = load_image(img_path)
    preds = model.predict(img)
    predicted_class = np.argmax(preds)
    print(f"Predicted class: {predicted_class}")

image_path = './test02.png'

model_path = './traffic_sign_identifier.keras'
recognize_image(image_path, model_path)
