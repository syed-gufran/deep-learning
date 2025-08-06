import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{dir}/model/trained_fahion_mnist_model.h5"

model = tf.keras.models.load_model(model_path)

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L') 
    img = np.array(img) /255.0
    img = img.reshape(1, 28, 28, 1)
    return img


st.title("Fashion MNIST Classifier")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg" , "jpeg", "png"])

if uploaded_image is not None:
    image  = Image.open(uploaded_image)
    col1 , col2 = st.columns(2)

    with col1:
        resized_image = image.resize((100, 100))
        st.image(resized_image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button("Classify"):
            img_array = preprocess_image(uploaded_image)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            prediction = class_names[predicted_class]
            st.success(f"Predicted Class: {prediction}")

