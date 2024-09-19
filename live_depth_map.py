import cv2
import torch
import time
import numpy as np



# Load a MiDas model for depth estimation
model_type = "MiDaS_small"  # can use DPT_Hybrid or DPT_Large for better accuracy

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform # for dpt_large and dpt_hybrid use dpt_transform

# Open up the video capture from inbuilt camera
cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, img = cap.read() #reading from the camera
    start = time.time() 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # input transforms
    input_batch = transform(img).to(device)

    # Prediction 
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F) # basic min max normalization to display

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img) #for comparison
    cv2.imshow('Depth Map', depth_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

