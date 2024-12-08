import cv2
import numpy as np

def count_fingers(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a binary threshold to segment the hand from the background
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return 0 fingers
    if len(contours) == 0:
        return 0

    # Get the largest contour (the hand)
    largest_contour = max(contours, key=cv2.contourArea)

    # Check if the contour is large enough to calculate defects
    if len(largest_contour) < 5:
        return 0

    # Approximate the contour to remove noise (optional)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Create a convex hull around the largest contour
    hull = cv2.convexHull(approx_contour)

    # Check if the convex hull is valid
    if len(hull) == 0:
        return 0

    # Calculate convexity defects
    defects = cv2.convexityDefects(largest_contour, hull)

    # If no defects, return 0 fingers
    if defects is None:
        return 0

    # Count defects (representing fingers)
    finger_count = 0
    for defect in defects:
        s, e, f, d = defect[0]
        if d > 2000:  # Adjust this value to suit your application
            finger_count += 1

    # If there are defects, we have fingers
    if finger_count > 0:
        return finger_count + 1  # Adding 1 for the palm
    else:
        return 0

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Detect the number of fingers in the frame
    fingers = count_fingers(frame)

    # Display the resulting frame with finger count
    cv2.putText(frame, f'Fingers: {fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Finger Count', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()
