import pickle
import cv2
import numpy as np
import mediapipe as mp

# Load your trained model
model_dict = pickle.load(open('./model2.p', 'rb'))
model = model_dict['model']

# Load a static image
image_path = 'trial3.jpg'  # Specify your image path
frame = cv2.imread(image_path)
labels_dict = {0: 'A', 1:'B', 2:'C'}
if frame is None:
    print("Error: Could not load image.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Prepare data
data_aux = []
x_ = []
y_ = []

# Convert image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect hands
results = hands.process(frame_rgb)

# Check if hands are detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract landmarks for prediction
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

    # Pad data_aux to ensure it has exactly 100 features
    while len(data_aux) < 100:
        data_aux.append(0)

    # Make prediction
    if len(data_aux) == 100:  # Ensure data_aux has 100 features
        prediction = model.predict([np.asarray(data_aux)])
        print("Prediction:", prediction)  # Print prediction to debug

        # Convert the prediction to an integer
        predicted_index = int(prediction[0])

        if predicted_index in labels_dict:
            predicted_character = labels_dict[predicted_index]
        else:
            predicted_character = "Unknown"  # Handle cases where prediction is out of bounds
    else:
        predicted_character = "Invalid Input"  # If not enough features

    # Display result
    cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)



else:
    print("No hands detected in the image.")

cv2.imshow('Hand Gesture Prediction', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
