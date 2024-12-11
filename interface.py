import gradio as gr
import pickle
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the model
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Create the dictionary for label mapping
label_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
              12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: "X", 
              23: 'Y', 24: '1', 25: '2', 26: '3', 27: '4', 28: '5', 29: '6', 30: '7', 31: '8', 32: '9'}

# ORB feature extractor initialization
orb = cv2.ORB_create()

# Function to extract features from an image
def extract_features(image):
    # Convert image to grayscale (if it's not already)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    
    if descriptors is not None:
        # Use KMeans clustering to get the feature histogram (like during training)
        # Load the kmeans model (same number of clusters used during training)
        kmeans = pickle.load(open('kmeans_model.p', 'rb'))  # Make sure this file exists and contains your k-means model
        
        cluster_indices = kmeans.predict(descriptors)
        histogram = np.histogram(cluster_indices, bins=np.arange(50 + 1))[0] 
        histogram = histogram.astype(float) / np.sum(histogram) if np.sum(histogram) > 0 else histogram
        return histogram
    else:
        return np.zeros(50)  # Return a zero vector if no descriptors are found

# Prediction function
def predict_sign_language(image):
    features = extract_features(image)  # Extract features from the input image
    print("Extracted features:", features)
    prediction = model.predict([features])  # Predict the class
    label = prediction[0]  # Get the predicted label (class index)
    
    # Map the class index to its corresponding label (letter/number)
    predicted_label = label_dict.get(label, 'Unknown')
    
    return predicted_label

# Create Gradio interface
gr.Interface(fn=predict_sign_language, inputs=gr.Image(type="numpy"), outputs="text").launch()
