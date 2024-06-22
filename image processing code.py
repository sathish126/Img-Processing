import numpy as np
import cv2

# Define known parameters
real_ball_diameter = 0.035  # Diameter of the ball in meters (10 cm)
focal_length = 1000        # Focal length of the camera in pixels (example value)

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

while True:
    # Reading the video from the webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame from BGR (RGB color space) to HSV (hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for yellow color and define mask
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Morphological Transform, Dilation for yellow color
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel)

    # Find contours
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles around contours and calculate distance
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            # Calculate the radius of the enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(imageFrame, center, radius, (0, 255, 255), 2)
            cv2.putText(imageFrame, "Yellow Ball", center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

            # Calculate distance using the simplified formula
            apparent_diameter = radius * 2  # Apparent diameter of the ball in pixels
            distance = (real_ball_diameter * focal_length) / apparent_diameter
            cv2.putText(imageFrame, f"Distance: {distance:.2f} meters", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Displaying the output
    cv2.imshow("Yellow Ball Detection in Real-Time", imageFrame)

    # Exiting the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()
