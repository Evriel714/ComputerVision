import gradio as gr
import pickle
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

orb = cv2.ORB_create()
knn = KNeighborsClassifier(n_neighbors=5)

path_train = "data/train/"
path_test = "data/test/"

data = []
gestures = []
gestures_test = []
best_matches = 0 
for folder in os.listdir(path_train):
    descriptor_list = []
    for img_path in os.listdir(path_train + folder):
        currpath = path_train + folder + '/' + img_path
        # print(currpath)
        img = cv2.imread(currpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.medianBlur(processed_img, 3)
        processed_img = cv2.equalizeHist(processed_img)
        img_keypoint, img_descriptor = orb.detectAndCompute(processed_img, None)
        # img_descriptor = np.float32(img_descriptor)
        descriptor_list.append(img_descriptor)
    gestures.append((folder, np.vstack(descriptor_list)))

for folder in os.listdir(path_test):
    descriptor_list = []
    for img_path in os.listdir(path_test + folder):
        currpath = path_test + folder + '/' + img_path
        # print(currpath)
        img = cv2.imread(currpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.medianBlur(processed_img, 3)
        processed_img = cv2.equalizeHist(processed_img)
        img_keypoint, img_descriptor = orb.detectAndCompute(processed_img, None)
        # img_descriptor = np.float32(img_descriptor)
        descriptor_list.append(img_descriptor)
    gestures_test.append((folder, np.vstack(descriptor_list)))

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# Prepare data for the classifier
x_train = []
y_train = []

x_test = []
y_test = []


for gesture_id, descriptors in gestures:
    for descriptor in descriptors:
        x_train.append(descriptor)
        y_train.append(int(gesture_id))

for gesture_id, descriptors in gestures_test:
    for descriptor in descriptors:
        x_test.append(descriptor)
        y_test.append(int(gesture_id))

x_train = np.array(x_train, dtype=np.float32) 
y_train = np.array(y_train)

x_test = np.array(x_test, dtype=np.float32) 
y_test = np.array(y_test)

pca = PCA(n_components=0.95)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)
# random_forest = RandomForestClassifier()
knn.fit(x_train_pca, y_train)


gesture_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 
                   10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 
                   19: 'U', 20: 'V', 21: 'W', 22: "X", 23: 'Y', 24: '1', 25: '2', 26: '3', 27: '4', 
                   28: '5', 29: '6', 30: '7', 31: '8', 32: '9'}

def recognize_gesture(image):
    # Ensure the image is a NumPy array (Gradio provides this automatically for type="numpy")
    if isinstance(image, str):  # If provided as filepath, read the image
        image = cv2.imread(image)
    
    target = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    process_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    process_target = cv2.medianBlur(process_target, 3)
    process_target = cv2.equalizeHist(process_target)
    keypoints, descriptors = orb.detectAndCompute(process_target, None)

    if descriptors is not None:
        predictions = knn.predict(descriptors)
        gesture = np.bincount(predictions).argmax()
        return gesture_mapping[gesture]
    else:
        return "No keypoints found"

def process_image(image):
    recognized_gesture = recognize_gesture(image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Recognized gesture: {recognized_gesture}")
    plt.axis('off')
    plt.show()
    return recognized_gesture

# Gradio interface
demo = gr.Interface(fn=process_image, inputs=gr.Image(type="numpy"), outputs="text")
demo.launch()
