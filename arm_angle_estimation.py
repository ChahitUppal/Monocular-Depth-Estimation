import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    a = np.array(point1)  # First point (shoulder or wrist)
    b = np.array(point2)  # Elbow (vertex)
    c = np.array(point3)  # Third point (wrist or shoulder)

    # Calculate the angle using the cosine rule
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    
    return angle

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

        # Draw pose landmarks on the frame and calculate elbow angles
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates of relevant points for right and left arms
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate elbow angles
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Draw pose landmarks for arms only
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            # Display the calculated elbow angles
            cv2.putText(frame, f'Right Elbow Angle: {int(right_elbow_angle)}', 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f'Left Elbow Angle: {int(left_elbow_angle)}', 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the output frame
        cv2.imshow('Pose Detection - Elbow Angles', frame)

        # Break the loop if 'ESC' is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()