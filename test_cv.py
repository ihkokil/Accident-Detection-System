import cv2
import numpy as np

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)
cv2.imshow('Test Image', img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
