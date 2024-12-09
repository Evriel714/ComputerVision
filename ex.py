import numpy as np
import cv2

# Read the query image and train image in color (RGB)
query_img = cv2.imread('0.jpg')
train_img = cv2.imread('112.jpg')

# Convert both images to grayscale for Canny edge detection
query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
query_edges = cv2.Canny(query_img_bw, 10, 200)
train_edges = cv2.Canny(train_img_bw, 10, 200)



# Initialize the ORB detector algorithm
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors on the Canny edge images
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_edges, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_edges, None)

# query_img_with_keypoints = cv2.drawKeypoints(query_edges, queryKeypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# train_img_with_keypoints = cv2.drawKeypoints(train_edges, trainKeypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("Query Image Keypoints", query_img_with_keypoints)
# cv2.imshow("Train Image Keypoints", train_img_with_keypoints)

# Initialize the Matcher for matching keypoints
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

# Use KNN to get the two best matches for each descriptor
knn_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

# Apply Lowe's ratio test to filter out weak matches
good_matches = []
ratio_threshold = 0.75  # Lowe's recommended ratio is typically 0.75

for m, n in knn_matches:
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)

# Draw only the good matches, but on the original RGB images
final_img = cv2.drawMatches(query_edges, queryKeypoints,
                            train_edges, trainKeypoints,
                            good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Resize the final image to fit on screen
final_img = cv2.resize(final_img, (1000, 650))

# Show the final image with matches
cv2.imshow("Good Matches", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
