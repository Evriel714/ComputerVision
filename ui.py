import pickle
import cv2
import numpy as np
import gradio as gr
import mediapipe as mp

# Load Mediapipe model
model_dict = pickle.load(open('model.p', 'rb'))
mediapipe_model = model_dict['model']

# Load ORB model
with open('model_orb.p', 'rb') as f:
    orb_model = pickle.load(f)

# Load SIFT model
with open('model_sift.p', 'rb') as f:
    sift_model = pickle.load(f)

# Gesture mapping
gesture_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
                   10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
                   19: 'U', 20: 'V', 21: 'W', 22: "X", 23: 'Y', 24: '1', 25: '2', 26: '3', 27: '4',
                   28: '5', 29: '6', 30: '7', 31: '8', 32: '9'}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# Mediapipe prediction function
def process_mediapipe(image):
    
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W, _ = image.shape
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract normalized landmark positions
            data_aux = []
            x_ = []
            y_ = []

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

            # Ensure feature length is correct
            if len(data_aux) == 42:
                prediction = mediapipe_model.predict([np.asarray(data_aux)])
                predicted_character = gesture_mapping[int(prediction[0])]
                return predicted_character

    # Return a default message if no hands are detected
    return "No hand detected or invalid input."

orb = cv2.ORB_create()
sift = cv2.SIFT_create()
clusters = 50  # Assume this matches your trained model
kmeans_orb = pickle.load(open('kmeans_model_orb.p', 'rb'))
scaler_orb = pickle.load(open('scaler_model_orb.p', 'rb'))
kmeans_sift = pickle.load(open('kmeans_model_sift.p', 'rb'))
scaler_sift = pickle.load(open('scaler_model_sift.p', 'rb'))

# ORB prediction function
def predict_orb(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ensure correct color format
    process_target = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    process_target = cv2.medianBlur(process_target, 3)
    process_target = cv2.equalizeHist(process_target)
    keypoints, descriptors = orb.detectAndCompute(process_target, None)

    if descriptors is not None:
        histogram = np.zeros(clusters)
        cluster_assignments = kmeans_orb.predict(descriptors)
        for cluster_id in cluster_assignments:
            histogram[cluster_id] += 1
        histogram = scaler_orb.transform([histogram])
        prediction = orb_model.predict(histogram)
        return gesture_mapping[int(prediction[0])]

    return "No keypoints found or invalid input."

# SIFT prediction function
def predict_sift(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Ensure correct color format
    process_target = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    process_target = cv2.medianBlur(process_target, 3)
    process_target = cv2.equalizeHist(process_target)
    keypoints, descriptors = sift.detectAndCompute(process_target, None)

    if descriptors is not None:
        histogram = np.zeros(clusters)
        cluster_assignments = kmeans_sift.predict(descriptors)
        for cluster_id in cluster_assignments:
            histogram[cluster_id] += 1
        histogram = scaler_sift.transform([histogram])
        prediction = sift_model.predict(histogram)
        return gesture_mapping[int(prediction[0])]

    return "No keypoints found or invalid input."

# Gradio processing function
def process_image(method, image):
    if method == "ORB":
        prediction = predict_orb(image)
    elif method == "SIFT":
        prediction = predict_sift(image)
    elif method == "Mediapipe":
        prediction = process_mediapipe(image)
    else:
        prediction = "Invalid method"
    return image, prediction

# Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Dropdown(choices=["ORB", "SIFT", "Mediapipe"], label="Choose a Method"),
        gr.Image(type="numpy", label="Upload an Image")
    ],
    outputs=[
        gr.Image(label="Input Image"),
        gr.Text(label="Recognized Gesture")
    ],
    title="Gesture Recognition",
    description="Choose a method (ORB, SIFT, or Mediapipe) and upload an image. The recognized gesture will be displayed."
)

demo.launch()
