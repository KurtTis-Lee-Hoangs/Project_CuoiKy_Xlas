import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import os

def open_image(image_file):
    return Image.open(image_file)
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def predict(model, image):
    image = cv2.resize(image, (32, 32))
    image = preprocessing(image)
    predictions = model.predict(np.array([image]))  # Assuming your model expects a batch of images
    class_index = np.argmax(predictions)
    class_names = {0: 'Stop', 1: 'Right', 2: 'Left', 3: 'Straight'}
    predicted_class = class_names[class_index]
    return predicted_class

def run():
    st.subheader('Nhận Diện Biển Báo')
    if 'is_load' not in st.session_state:
        # load model
        model = load_model(f"{os.path.dirname(__file__)}/utility/NhanDienBienBao_Streamlit/my_trained_model.h5")
        st.session_state.model = model

        st.session_state.is_load = True
        print('Lần đầu load model')
    else:
        print('Đã load model rồi')

    image_file = st.file_uploader("Upload Images", type=["png", "jpg"])

    if (st.button('Open')):
        st.session_state.imageInn = open_image(image_file)
    if 'imageInn' in st.session_state:
        st.image(st.session_state.imageInn)

        if (st.button('Xử lý')):
            image_pil = Image.open(image_file)
            image_np = np.array(image_pil)
            st.session_state.imageOut = predict(st.session_state.model, image_np)
            st.text("Đây là biển báo: " + st.session_state.imageOut)

if __name__ == "__main__":
    run()
