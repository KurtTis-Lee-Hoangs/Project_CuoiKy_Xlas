import streamlit as st
import cv2
import mediapipe as mp

st.set_page_config(page_title="Nhận dạng ngón tay và đếm số ngón tay", page_icon="🖐️")
st.title("Nhận diện ngón tay và đếm số ngón tay")

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

                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img_rgb.shape
                    myHand.append([int(lm.x * w), int(lm.y * h)])

                for lm_index in fingersId:
                    if myHand[lm_index][1] < myHand[lm_index-2][1]:
                        count = count + 1  

                if myHand[4][0] < myHand[4-2][0] and myHand[5][0] <= myHand[13][0]:
                    count = count + 1  
                elif myHand[4][0] > myHand[4-2][0] and myHand[5][0] >= myHand[13][0]:
                    count = count + 1 

            cv2.putText(img_rgb, str(count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # Update the image in the placeholder
        image_placeholder.image(img_rgb, channels="BGR", use_column_width=True, caption="Face Recognition Result")

    # Release old image
    cap.release()

    # Close the OpenCV window when the Streamlit app stops
    cv2.destroyAllWindows()
    
st.button("Re-run")
