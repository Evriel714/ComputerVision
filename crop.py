import cv2
import mediapipe as mp
import numpy as np  # Import numpy for creating blank images

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)

image = cv2.imread('1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        h, w, _ = image.shape

        # Get the bounding box of the hand
        x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
        y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
        x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
        y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

        # Calculate the current width and height
        current_width = x_max - x_min
        current_height = y_max - y_min

        # Calculate the desired size
        desired_size = 240

        # Calculate the padding required to achieve the desired size
        padding_x = max(0, (desired_size - current_width) // 2)
        padding_y = max(0, (desired_size - current_height) // 2)

        # Adjust bounding box with padding
        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(h, y_max + padding_y)

        # Crop the hand from the image
        cropped_hand = image[y_min:y_max, x_min:x_max]

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

        # Display the cropped hand
        cv2.imshow('Cropped Hand', cropped_hand)
        cv2.waitKey(0)

cv2.destroyAllWindows()
