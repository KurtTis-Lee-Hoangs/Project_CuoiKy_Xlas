import cv2
import math
import streamlit as st
import mediapipe as mp
import win32gui, win32api, win32con

st.set_page_config(page_title="Nhận dạng ngón tay và di chuyển và click chuột bằng ngón tay", page_icon="✊")
st.title("Nhận dạng ngón tay và di chuyển và click chuột bằng ngón tay")

mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

start_button = st.button("Bắt đầu nhận diện")
stop_button = st.button("Dừng nhận diện")

oldPlace = []
newPlace = []

if start_button and not stop_button:
    cap = cv2.VideoCapture(0)

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    fingersId = [8, 12, 16, 20]
    image_placeholder = st.empty()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)

        # Convert the image to RGB before processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            myHand = []
            count = 0

            for idx, hand in enumerate(result.multi_hand_landmarks):
                mp_drawing_util.draw_landmarks(
                    img_rgb,
                    hand,
                    mp_hand.HAND_CONNECTIONS,
                    mp_drawing_style.get_default_hand_landmarks_style(),
                    mp_drawing_style.get_default_hand_connections_style()
                )

                h, w, _ = img.shape
                cx4, cy4 = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
                cx5, cy5 = int(hand.landmark[5].x * w), int(hand.landmark[5].y * h)
                cx8, cy8 = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
                
                oldPlace = newPlace.copy()
                if cy8 < cy5:
                    newPlace.clear()

                    newPlace.append(cx8)
                    newPlace.append(cy8)

                    newPlace.append(cx4)
                    newPlace.append(cy4)

                if len(oldPlace) > 0:
                    dr = int(math.sqrt((newPlace[0]-newPlace[2])**2 + (newPlace[1]-newPlace[3])**2))
                    if dr < 25:
                        flags, hcursor, (x,y) = win32gui.GetCursorInfo()
                        win32api.SetCursorPos((x,y))
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
                
                if len(oldPlace) > 0:
                    dx = newPlace[0] - oldPlace[0]
                    dy = newPlace[1] - oldPlace[1]

                    flags, hcursor, (x,y) = win32gui.GetCursorInfo()
                    win32api.SetCursorPos((x+dx*2,y+dy*2))

        # Update the image in the placeholder
        image_placeholder.image(img_rgb, channels="BGR", use_column_width=True, caption="Face Recognition Result")

    # Release old image
    cap.release()

    # Close the OpenCV window when the Streamlit app stops
    cv2.destroyAllWindows()
    
st.button("Re-run")
