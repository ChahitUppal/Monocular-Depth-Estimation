import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

# Initialize Pose detection model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB as Mediapipe uses RGB images
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image and perform pose detection
        results = pose.process(img_rgb)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        # Display the output frame
        cv2.imshow('Pose Detection', frame)

        # Break the loop if 'ESC' is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()