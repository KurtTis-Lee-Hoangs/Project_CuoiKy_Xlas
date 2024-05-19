import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import os

st.set_page_config(
    page_title="Trang Chủ",
    page_icon="👋",
)

logo = Image.open('images\logo.jpg')
st.image(logo, width=800)

st.markdown(
    """
    ### Website cuối kỳ của môn học: Xử Lý Ảnh Số
    - Thực hiện bởi: Lê Minh Hoàng và Nguyễn Trọng Phúc
    - Giảng viên hướng dẫn: ThS. Trần Tiến Đức
    - Lớp Xử Lý Ảnh Số nhóm 03: DIPR430685_23_2_03
    """
)

st.markdown("""### Thành viên thực hiện""")
left, right = st.columns(2)
with left: 
    st.image(Image.open(f"{os.path.dirname(__file__)}/images/minhhoang.jpg"), "Lê Minh Hoàng - 21110457", width=300)
with right:
    st.image(Image.open(f"{os.path.dirname(__file__)}/images/trongphuc.jpg"), "Nguyễn Trọng Phúc - 21110915", width=300)

st.markdown(
    """
    ### Thông tin liên hệ
    - Facebook: [Minh Hoàng](https://www.facebook.com/profile.php?id=100028798721439) hoặc [Trọng Phúc](https://www.facebook.com/profile.php?id=100045860234345)
    - Email: 21110457@student.hcmute.edu.vn hoặc 21110915@student.hcmute.edu.vn
    - Lấy source code tại: [đây](https://github.com/KurtTis-Lee-Hoangs/Project_CuoiKy_Xlas)
    """
)

st.markdown("""### Video giới thiệu về Website""")
st.markdown("""[Video giới thiệu website cuối kỳ môn học Xử Lý Ảnh Số]()""")