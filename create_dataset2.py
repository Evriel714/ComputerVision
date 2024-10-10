import os
import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans

DATA_DIR = './data'

# ORB Feature extractor initialization
orb = cv2.ORB_create()

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Store keypoints and descriptors from all images
keypoints_list = []
descriptors_list = []
image_paths = []

# Iterate through each folder to extract keypoints and descriptors
for class_folder in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_folder)

    # Iterate through each image in the class folder
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Extract keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        if descriptors is not None:
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
            image_paths.append(img_path)

# Combine all descriptors for clustering
all_descriptors = np.vstack(descriptors_list)

# Perform K-means clustering
num_clusters = 100  # Adjust this based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(all_descriptors)

# Create a feature vector for each image
feature_vectors = []
for descriptors in descriptors_list:
    # Get the cluster indices for the current image descriptors
    cluster_indices = kmeans.predict(descriptors)
    
    # Create a histogram of clusters (feature vector)
    histogram = np.histogram(cluster_indices, bins=np.arange(num_clusters + 1))[0]
    
    # Normalize the histogram
    histogram = histogram.astype(float) / np.sum(histogram) if np.sum(histogram) > 0 else histogram
    
    feature_vectors.append(histogram)

# Save the feature vectors and image paths to a pickle file
data_to_save = {
    'feature_vectors': feature_vectors,
    'image_paths': image_paths,
}

with open('orb_sign_language_features.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)