# import os
# import cv2


# DATA_DIR = './data'
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# number_of_classes = 2
# dataset_size = 50

# cap = cv2.VideoCapture(0)
# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
#         os.makedirs(os.path.join(DATA_DIR, str(j)))

#     print('Collecting data for class {}'.format(j))

#     done = False
#     while True:
#         ret, frame = cap.read()
#         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
#                     cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(25) == ord('q'):
#             break

#     counter = 0
#     while counter < dataset_size:
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(25)

#         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

#         flipped_frame = cv2.flip(frame, 1)
#         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter * 2 + 1)), flipped_frame)

#         counter += 1

# cap.release()
# cv2.destroyAllWindows()


import os
import cv2
import mediapipe as mp
import numpy as np  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 33
dataset_size = 50

# target = 33

cap = cv2.VideoCapture(1)
for j in range(1, number_of_classes):
    # if j!=target:
    #     continue

    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Wait for user to press 'Q' to start collecting data
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    image_num = 300
    while counter < dataset_size:
        ret, frame = cap.read()

        # Convert frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape

                # Get the bounding box of the hand
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Calculate the current width and height
                current_width = x_max - x_min
                current_height = y_max - y_min

                # Calculate the desired size
                desired_size = 300

                # Calculate the padding required to achieve the desired size
                padding_x = max(0, (desired_size - current_width) // 2)
                padding_y = max(0, (desired_size - current_height) // 2)

                # Adjust bounding box with padding
                x_min = max(0, x_min - padding_x)
                x_max = min(w, x_max + padding_x)
                y_min = max(0, y_min - padding_y)
                y_max = min(h, y_max + padding_y)

                # Crop the hand from the image
                cropped_hand = frame[y_min:y_max, x_min:x_max]

                # Check if the cropped area needs to be adjusted
                cropped_height, cropped_width = cropped_hand.shape[:2]

                # Ensure the cropped hand is exactly 240x240
                if cropped_height < desired_size or cropped_width < desired_size:
                    # Create a blank image with the desired size
                    blank_image = 255 * np.ones((desired_size, desired_size, 3), dtype=np.uint8)

                    # Calculate offsets to center the cropped hand in the blank image
                    y_offset = (desired_size - cropped_height) // 2
                    x_offset = (desired_size - cropped_width) // 2

                    # Place the cropped image at the center of the blank image
                    blank_image[y_offset:y_offset + cropped_height, x_offset:x_offset + cropped_width] = cropped_hand
                    cropped_hand = blank_image
                else:
                    # Crop the image to 240x240 if it's overshot
                    cropped_hand = cropped_hand[:desired_size, :desired_size]

                # Save the cropped image
                cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(image_num)), cropped_hand)
                image_num += 1

                # Flip the image horizontally and save
                flipped_hand = cv2.flip(cropped_hand, 1)
                cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(image_num)), flipped_hand)
                image_num += 1

                counter += 1

        # Show the current frame
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()