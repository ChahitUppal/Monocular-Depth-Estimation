import cv2
import torch
import time
import numpy as np
import mediapipe as mp
import math

# Load a MiDaS model for depth estimation
model_type = "MiDaS_small"  # can use DPT_Hybrid or DPT_Large for better accuracy
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # for dpt_large and dpt_hybrid use dpt_transform

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Depth to distance conversion
def depth_to_distance(depth):
    return -1.7 * depth + 2

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Second point (elbow)
    c = np.array(c)  # Third point (wrist)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    
    return angle

# Open up the video capture from inbuilt camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()  # reading from the camera
    if not success:
        break
    
    start = time.time()
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pose detection with MediaPipe
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract the key points
        left_shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * img.shape[1]), 
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * img.shape[0])]
        left_elbow = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * img.shape[1]), 
                      int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * img.shape[0])]
        left_wrist = [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * img.shape[1]), 
                      int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * img.shape[0])]
        
        right_shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img.shape[1]), 
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img.shape[0])]
        right_elbow = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * img.shape[1]), 
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * img.shape[0])]
        right_wrist = [int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * img.shape[1]), 
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * img.shape[0])]
        
        # Calculate the elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Display the elbow angles on the image
        cv2.putText(img, f"Left Elbow Angle: {int(left_elbow_angle)}", (left_elbow[0] - 50, left_elbow[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(img, f"Right Elbow Angle: {int(right_elbow_angle)}", (right_elbow[0] - 50, right_elbow[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Transform image for depth estimation
        input_batch = transform(img_rgb).to(device)

        # Depth prediction
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        # Get the depth at the elbows
        if 0 <= left_elbow[0] < depth_map.shape[1] and 0 <= left_elbow[1] < depth_map.shape[0]:
            left_elbow_depth = depth_map[left_elbow[1], left_elbow[0]]
            left_elbow_distance = depth_to_distance(left_elbow_depth)
            cv2.putText(img, f"Left Elbow Depth: {left_elbow_distance:.2f}m", (left_elbow[0], left_elbow[1] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        if 0 <= right_elbow[0] < depth_map.shape[1] and 0 <= right_elbow[1] < depth_map.shape[0]:
            right_elbow_depth = depth_map[right_elbow[1], right_elbow[0]]
            right_elbow_distance = depth_to_distance(right_elbow_depth)
            cv2.putText(img, f"Right Elbow Depth: {right_elbow_distance:.2f}m", (right_elbow[0], right_elbow[1] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Display the depth map
        depth_map_visual = (depth_map * 255).astype(np.uint8)
        depth_map_visual = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)

        # Show the FPS
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Display the images
        cv2.imshow('Pose and Depth Estimation', img)
        cv2.imshow('Depth Map', depth_map_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()