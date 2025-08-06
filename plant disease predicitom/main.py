import os
from PIL import Image
import streamlit as st
import numpy as np
import tensorflow as tf
import json

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'plant_disease_prediction_model.h5')

model = tf.keras.models.load_model(model_path)

class_indeices = json.load(open(os.path.join(working_dir, 'class_indices.json')))

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img)   # Normalize the image
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array.astype('float32') / 255.0  # Add batch dimension
    return img_array

def predict_disease(model , image_path , class_indeices):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_indeices[str(predicted_class_index)]
    return predicted_class

st.title("Plant disease classifier")

uploaded_img = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    col1 , col2 = st.columns(2)

    with col1:
        resized_image = image.resize((224, 224))
        st.image(resized_image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button("Classify"):
            with st.spinner("Predicting..."):
                predicted_class = predict_disease(model , uploaded_img , class_indeices)
                st.success(f"The plant disease is: {str(predicted_class)}")