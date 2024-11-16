import cv2
import torch
from fastai.tabular import model
from fastai.vision.all import load_learner
from torch import nn
from torchvision.models import resnet34
from ultralytics import YOLO
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path

# Define the project paths
project_dir = Path(__file__).resolve().parent.parent
model_dir = project_dir / 'Model'
data_dir = project_dir / 'Data'

sys.path.append(str(project_dir / '1_HaGRID'))
sys.path.append(str(project_dir / '2_Modelling' / 'Transforms'))

import constants

# Load YOLO model
yolo_model_path = model_dir / "YOLOv10x_hands.pt"
model_yolo = YOLO(yolo_model_path)

# Load ResNet model
resnet_model_path = model_dir / "resnet34_unfrozen_best.pth"
resnet_model = resnet34()
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(constants.targets))

resnet_model.load_state_dict(torch.load(resnet_model_path, weights_only=True))
resnet_model.eval()

# Define targets (same as during training)
gesture_targets = list(constants.targets.values())  # Add your gestures


# Function to crop and preprocess the hand region
def crop_and_transform(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Convert to PIL Image
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    # Resize to 224x224
    cropped_image = F.resize(cropped_image, [224, 224])

    # Normalize (using the same mean and std as in training)
    cropped_image = F.to_tensor(cropped_image)
    cropped_image = F.normalize(cropped_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return cropped_image.unsqueeze(0)  # Add batch dimension


# Function for real-time hand gesture recognition
def hand_gesture_recognition():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model to detect hands
        results = model_yolo(frame)

        for result in results:
            for detection in result.boxes:
                bbox = detection.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Crop and transform the image
                cropped_image = crop_and_transform(frame, bbox)

                # Predict gesture using ResNet
                with torch.no_grad():
                    output = resnet_model(cropped_image)
                    predicted_class = torch.argmax(output, dim=1).item()
                    gesture_label = gesture_targets[predicted_class]

                # Display the bounding box and predicted gesture
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, gesture_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the frame
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the real-time recognition
hand_gesture_recognition()
