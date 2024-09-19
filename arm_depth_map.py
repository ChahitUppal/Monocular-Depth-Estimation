import cv2
import torch
import time
import numpy as np
import mediapipe as mp

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
        # Get the coordinates of the left and right elbows
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        left_elbow_x, left_elbow_y = int(left_elbow.x * img.shape[1]), int(left_elbow.y * img.shape[0])
        right_elbow_x, right_elbow_y = int(right_elbow.x * img.shape[1]), int(right_elbow.y * img.shape[0])

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

        # Ensure the elbow coordinates are within the depth map's range
        if 0 <= left_elbow_x < depth_map.shape[1] and 0 <= left_elbow_y < depth_map.shape[0]:
            left_elbow_depth = depth_map[left_elbow_y, left_elbow_x]
            left_elbow_distance = depth_to_distance(left_elbow_depth)
            cv2.putText(img, f"Left Elbow Depth: {left_elbow_distance:.2f}m", (left_elbow_x, left_elbow_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if 0 <= right_elbow_x < depth_map.shape[1] and 0 <= right_elbow_y < depth_map.shape[0]:
            right_elbow_depth = depth_map[right_elbow_y, right_elbow_x]
            right_elbow_distance = depth_to_distance(right_elbow_depth)
            cv2.putText(img, f"Right Elbow Depth: {right_elbow_distance:.2f}m", (right_elbow_x, right_elbow_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the depth map
        depth_map_visual = (depth_map * 255).astype(np.uint8)
        depth_map_visual = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('Image', img)  # for comparison
        cv2.imshow('Depth Map', depth_map_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()