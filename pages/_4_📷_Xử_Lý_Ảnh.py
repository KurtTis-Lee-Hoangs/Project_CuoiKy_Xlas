import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utility/')))

from XuLyAnh_Streamlit import Chapter03 as c3
from XuLyAnh_Streamlit import Chapter04 as c4
from XuLyAnh_Streamlit import Chapter05 as c5
from XuLyAnh_Streamlit import Chapter09 as c9


def open_image(image_file):
    return Image.open(image_file)

def process_image(selected_option, image_in):
    img_array = np.array(image_in)

    if selected_option == "Negative":
        return Image.fromarray(cv2.cvtColor(c3.Negative(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "Logarit":
        return Image.fromarray(cv2.cvtColor(c3.Logarit(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "PiecewiseLinear":
        return Image.fromarray(cv2.cvtColor(c3.PiecewiseLinear(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "Histogram":
        return Image.fromarray(cv2.cvtColor(c3.Histogram(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "HistEqual":
        return Image.fromarray(cv2.cvtColor(c3.HistEqual(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "HistEqualColor":
        return Image.fromarray(cv2.cvtColor(c3.HistEqualColor(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "LocalHist":
        return Image.fromarray(cv2.cvtColor(c3.LocalHist(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "HistStat":
        return Image.fromarray(cv2.cvtColor(c3.HistStat(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "BoxFilter":
        return Image.fromarray(cv2.cvtColor(cv2.blur(img_array, (21, 21)), cv2.COLOR_BGR2RGB))
    elif selected_option == "LowpassGauss":
        return Image.fromarray(cv2.cvtColor(cv2.GaussianBlur(img_array, (43, 43), 7.0), cv2.COLOR_BGR2RGB))
    elif selected_option == "Threshold":
        return Image.fromarray(cv2.cvtColor(c3.Threshold(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "MedianFilter":
        return Image.fromarray(cv2.cvtColor(cv2.medianBlur(img_array, 7), cv2.COLOR_BGR2RGB))
    elif selected_option == "Sharpen":
        return Image.fromarray(cv2.cvtColor(c3.Sharpen(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "Gradient":
        return Image.fromarray(cv2.cvtColor(c3.Gradient(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "Spectrum":
        return Image.fromarray(cv2.cvtColor(c4.Spectrum(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "FrequencyFilter":
        return Image.fromarray(cv2.cvtColor(c4.FrequencyFilter(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "DrawNotchRejectFilter":
        return Image.fromarray(cv2.cvtColor(c4.DrawNotchRejectFilter(), cv2.COLOR_BGR2RGB))
    elif selected_option == "RemoveMoire":
        return Image.fromarray(cv2.cvtColor(c4.RemoveMoire(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "CreateMotionNoise":
        return Image.fromarray(cv2.cvtColor(c5.CreateMotionNoise(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "DenoiseMotion":
        return Image.fromarray(cv2.cvtColor(c5.DenoiseMotion(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "DenoisestMotion":
        temp = cv2.medianBlur(img_array, 7)
        return Image.fromarray(cv2.cvtColor(c5.DenoiseMotion(temp), cv2.COLOR_BGR2RGB))
    elif selected_option == "Erosion":
        imgout = np.array(image_in)
        c9.Erosion(img_array, imgout)
        return Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    elif selected_option == "Dilation":
        imgout = np.array(image_in)
        c9.Dilation(img_array, imgout)
        return Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    elif selected_option == "OpeningClosing":
        imgout = np.array(image_in)
        c9.OpeningClosing(img_array, imgout)
        return Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    elif selected_option == "Boundary":
        return Image.fromarray(cv2.cvtColor(c9.Boundary(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "ConnectedComponent":
        return Image.fromarray(cv2.cvtColor(c9.ConnectedComponent(img_array), cv2.COLOR_BGR2RGB))
    elif selected_option == "CountRice":
        return Image.fromarray(cv2.cvtColor(c9.CountRice(img_array), cv2.COLOR_BGR2RGB))


def run():
    #st.set_page_config(page_title="Xử lý ảnh", page_icon=":camera:")
    st.title("Ứng dụng Xử lý ảnh")

    options_1 = ['Open', 'OpenColor', 'Save', 'Exit']
    options_2 = ['Chương 3', 'Chương 4', 'Chương 5', 'Chương 9']

    if 'options_3' not in st.session_state:
        st.session_state.options_3 = ['Negative', 'Logarit', 'PiecewiseLinear', 'Histogram', 'HistEqual', 'HistEqualColor',
                'LocalHist', 'HistStat', 'BoxFilter', 'LowpassGauss', 'Threshold', 'MedianFilter', 'Sharpen']
        print('Chua load')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button('Chương 3'):
            st.session_state.options_3 = ['Negative', 'Logarit', 'PiecewiseLinear', 'Histogram', 'HistEqual', 'HistEqualColor',
                'LocalHist', 'HistStat', 'BoxFilter', 'LowpassGauss', 'Threshold', 'MedianFilter', 'Sharpen', 'Gradient']
    with col2:
        if st.button('Chương 4'):
            st.session_state.options_3 = ['Spectrum', 'FrequencyFilter', 'DrawNotchRejectFilter', 'RemoveMoire']
    with col3:
        if st.button('Chương 5'):
            st.session_state.options_3 = ['CreateMotionNoise', 'DenoiseMotion', 'DenoisestMotion']
    with col4:
        if st.button('Chương 9'):
            st.session_state.options_3 = ['Erosion', 'Dilation', 'OpeningClosing', 'Boundary',
                'ConnectedComponent', 'CountRice']

    selected_option = st.selectbox('Chọn phương pháp xử lí ảnh:', st.session_state.options_3)

    image_file = st.file_uploader("Upload Images", type=["bmp", "png", "jpg", "jpeg", "tif"])

    if (st.button('Open')):
        st.session_state.imageIn = open_image(image_file)

    if 'imageIn' in st.session_state:
        st.image(st.session_state.imageIn)

        if (st.button('Xử lý')):
            st.session_state.imageOut = process_image(selected_option, st.session_state.imageIn)
            st.image(st.session_state.imageOut)

if __name__ == "__main__":
    run()
