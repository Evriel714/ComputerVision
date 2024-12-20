import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './test'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    # n = 0
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # if n == 60:
        #     break
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Process only the first hand
            hand_landmarks = results.multi_hand_landmarks[0]

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure only processed data is appended
            if len(data_aux) == 42:  # Validate feature length
                data.append(data_aux)
                labels.append(dir_)
        else:
            print(f"No hand detected in {img_path}. Skipping.")
        # n += 1

print(len(data))

# Save data to pickle
with open('test_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
