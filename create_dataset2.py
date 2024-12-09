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

keypoints_list = []
descriptors_list = []
labels = []  # Labels for each image

# Specify the class folders you want to iterate over
target_folders = ['0', '1', '2']  # Replace these with the folder names you want to process

# Iterate through the specified folders to extract keypoints and descriptors
for class_folder in target_folders:
    class_path = os.path.join(DATA_DIR, class_folder)

    # Check if the class folder exists
    if not os.path.exists(class_path):
        print(f"Class folder {class_path} does not exist. Skipping.")
        continue

    # Iterate through each image in the class folder
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue

        # Apply Canny edge detection with threshold 100-200
        edges = cv2.Canny(img, 100, 200)
        
        # Extract keypoints and descriptors using ORB on the Canny result
        keypoints, descriptors = orb.detectAndCompute(edges, None)
        
        if descriptors is not None:
            # Match descriptors using BFMatcher
            matches = bf.match(descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Set a distance threshold (lower is stricter, retain best matches)
            good_matches = [m for m in matches if m.distance < 40]  # Example threshold

            # Retain only good descriptors based on the good matches
            good_descriptors = np.array([descriptors[m.queryIdx] for m in good_matches])

            # Append if good descriptors exist
            if len(good_descriptors) > 0:
                keypoints_list.append(keypoints)
                descriptors_list.append(good_descriptors)
                labels.append(class_folder)  # Append the class folder name as the label

# Combine all descriptors for clustering
if len(descriptors_list) > 0:
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

    # Save the feature vectors and labels to a pickle file
    data_to_save = {
        'feature_vectors': feature_vectors,
        'labels': labels,  # Save labels instead of image paths
    }

    with open('orb_sign_language_features.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    print("Feature vector construction and saving completed!")
else:
    print("No descriptors were found.")
