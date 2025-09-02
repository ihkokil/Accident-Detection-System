import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5.keras')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video_path = 'cars.mp4'  # Use 0 for webcam
    video = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    accident_detected = False  # Flag to track if accident is detected

    while True:
        ret, frame = video.read()

        # Check if frame is read correctly
        if not ret or frame is None:
            print("Error: Could not read frame or end of video reached.")
            break

        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"Error during cvtColor: {e}")
            break

        roi = cv2.resize(gray_frame, (250, 250))

        try:
            pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        except Exception as e:
            print(f"Error during prediction: {e}")
            break

        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)

            # to beep when alert:
            # if prob > 90:
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}", (20, 30), font, 1, (255, 255, 0), 2)

            # Set accident_detected flag if accident is detected
            accident_detected = True

        cv2.imshow('Video', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Print result based on accident_detected flag
    if accident_detected:
        print("Accident detected")
    else:
        print("Accident not detected")

if __name__ == '__main__':
    startapplication()
