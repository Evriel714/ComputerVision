import os
import cv2
import matplotlib.pyplot as plt

# Path to your dataset
DATA_DIR = './data'

# ORB Feature extractor initialization
orb = cv2.ORB_create()

# Number of images to display per folder
num_images_to_display = 1

# Create a list to hold images for display
images_to_show = []

# Iterate through the specified folders (0, 1, 2)
for class_folder in ['0', '1', '2']:
    class_path = os.path.join(DATA_DIR, class_folder)

    # Counter to limit the number of images displayed
    count = 0

    # Iterate through each image in the class folder
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue
        
        # Extract keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(img, None)

        # Draw keypoints on the image
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Append the image with keypoints to the list for display
        images_to_show.append((class_folder, img_with_keypoints))
        
        count += 1
        if count >= num_images_to_display:
            break  # Stop after displaying the specified number of images per folder

# Set up the plot
plt.figure(figsize=(15, 5))

# Display the images with keypoints
for i, (class_label, image) in enumerate(images_to_show):
    plt.subplot(1, len(images_to_show), i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Class {class_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()
