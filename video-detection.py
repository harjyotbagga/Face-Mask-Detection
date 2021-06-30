import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
from imutils.video import VideoStream
import time
import warnings

warnings.filterwarnings("ignore")

vs = VideoStream(src=0).start()
time.sleep(1.0)
faceNet = cv2.dnn.readNetFromCaffe('Trained Models/face_detector/deploy.prototxt', 'Trained Models/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
THRESHOLD_CONFIDENCE = 0.5

trained_model = torchvision.models.mobilenet_v2(pretrained=True)
trained_model.classifier[1] = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(128, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 32),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(32, 2),
)
trained_model.load_state_dict(torch.load('Trained Models/mask_detection_model.pth'))
trained_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_transform_compose = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vs = VideoStream(src=0).start()
time.sleep(1.0)

def detect_mask(frame):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = torch.Tensor()
    boxes = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > THRESHOLD_CONFIDENCE:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            face = Image.fromarray(face)
            face = test_transform_compose(face)
            face = face.unsqueeze_(0)
            
            faces = torch.cat((faces, face), dim=0)
            boxes.append((startX, startY, endX, endY))
            
        if len(faces) > 0:
            faces = faces.to(device=device)
            trained_model.to(device=device)
            pred_output = trained_model(faces)
            _, preds = pred_output.max(1)
        
        return (boxes, preds)    
            
    return frame

while True:
    frame = vs.read()
    (boxes, preds)  = detect_mask(frame=frame)
    for (box, pred) in zip(boxes, preds):
        (startX, startY, endX, endY) = box
        color = (0,255,0) if pred == 1 else (0,0,255)
        label = "MASK" if pred == 1 else "NO MASK"
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX+10, endY+25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
