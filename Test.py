import cv2
import numpy as np
 
# Load the image
image = cv2.imread("Photos/mediumbolt.jpg")
cv2.imshow("original", image)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Apply Gaussian blur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
 
# Use adaptive thresholding to create a binary image
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 2)
 
# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 
# Loop over the contours
for contour in contours:
    # Compute the center of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:  # avoid division by zero error
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
 
    # Get contour area
    area = cv2.contourArea(contour)
 
    # Label the object based on area (this is a simplistic approach and can be enhanced)
    label = str(area)
    if 35300 > area > 30000:
        label = "Large Bolt"
    elif 7500 > area > 5800:
        label = "Medium Bolt"
    elif 4500 > area > 3800:
        label = "Small Bolt"
    elif 8500 > area > 7500:
        label = "Small Nut"
    elif 10200 > area > 9300:
        label = "Medium Nut"
    elif 20000 > area > 15000:
        label = "Large Nut"
 
   
 
    # Draw the contour and label on the image
    if area > 2500:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
# Show the labeled image
 
cv2.imshow("Grayscale", gray)
cv2.imshow("blurred", blurred)
cv2.imshow("binary", binary)
cv2.imshow("Labeled Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()