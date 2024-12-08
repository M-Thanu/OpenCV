import cv2
print("Hello")
# Read an image from a file (make sure the image path is correct)
image = cv2.imread('C:\\Users\\hp\\Downloads\\shuttle.jpeg')

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found")
else:
    # Display the image in a window
    cv2.imshow('Display Window', image)

    # Wait for a key press indefinitely or for a specific amount of time (in milliseconds)
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
