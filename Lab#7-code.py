import cv2
import matplotlib.pyplot as plt
import numpy as np
# Load the image
image = cv2.imread("C:\\Users\\zezva\\Desktop\\Lab#7\\cameraman.tif", cv2.IMREAD_GRAYSCALE)

# Create SIFT detector object
sift = cv2.SIFT_create()

# Detect SIFT features
keypoints, descriptors = sift.detectAndCompute(image, None)

# Draw the keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.figure(figsize=(10, 10))
plt.imshow(image_with_keypoints, cmap='gray')
plt.title('SIFT Keypoints')
plt.axis('off')
plt.show()

#############################  Display the 10 strongest points

# Sort keypoints based on their response (strength)
keypoints = sorted(keypoints, key=lambda x: -x.response)

# Sort keypoints first by response and then by x-coordinate
#keypoints = sorted(keypoints, key=lambda x: (x.pt[0], -x.response,))

# Display the 10 strongest points
strongest_keypoints = keypoints[:10]
image_with_strongest_keypoints = cv2.drawKeypoints(image, strongest_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Print the coordinates of the strongest keypoints
print("Coordinates of the 10 strongest keypoints:")
for i, kp in enumerate(strongest_keypoints):
    print(f"Point {i+1}: ({kp.pt[0]}, {kp.pt[1]})")

# Display the image with the 10 strongest keypoints
plt.figure(figsize=(10, 10))
plt.imshow(image_with_strongest_keypoints, cmap='gray')
plt.title('10 Strongest SIFT Keypoints')
plt.axis('off')
plt.show()

########################### Detect and display the last 5 SIFT features

# Display the last 5 detected points
last_keypoints = keypoints[-5:]
image_with_last_keypoints = cv2.drawKeypoints(image, last_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with the last 5 keypoints
plt.figure(figsize=(10, 10))
plt.imshow(image_with_last_keypoints, cmap='gray')
plt.title('Last 5 SIFT Keypoints')
plt.axis('off')
plt.show()

###########################   Detect Harris corner features and extract them

# Detect Harris corners
harris_corners = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)

# Dilate corner image to enhance corner points
harris_corners = cv2.dilate(harris_corners, None)

# Threshold for an optimal value, it may vary depending on the image
threshold = 0.01 * harris_corners.max()
image_with_harris = image.copy()
image_with_harris[harris_corners > threshold] = 255

# Display the Harris corners
plt.figure(figsize=(10, 10))
plt.imshow(image_with_harris, cmap='gray')
plt.title('Harris Corners')
plt.axis('off')
plt.show()

#########################   Extract Harris features:

# Find and extract corner features from the image
corners = cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

# Draw corners on the image
image_with_corners = image.copy()
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image_with_corners, (x, y), 3, 255, -1)

# Display the corners
plt.figure(figsize=(10, 10))
plt.imshow(image_with_corners, cmap='gray')
plt.title('Extracted Corner Features')
plt.axis('off')
plt.show()

