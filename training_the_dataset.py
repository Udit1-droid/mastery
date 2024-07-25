import multiprocessing
from ultralytics import YOLO
import os
import torch

device='cuda' if torch.cuda.is_available() else 'cpu'

# Set the paths to your YOLO directory structure
yolo_path = r'E:/chilika_vd/dataset'
data_yaml = r'E:/chilika_vd/dataset/p_data.yaml'

# Load a YOLOv8 model
model = YOLO('yolov8s.pt').to(device)  # You can choose a different YOLO model

if __name__=='__main__':
    multiprocessing.freeze_support()
# Train the model
    model.train(
        data=data_yaml,
        epochs=150,
        imgsz=384,
        batch=16,
        workers=4,
        name='training_run_yolov8sp',
        project=yolo_path,
        exist_ok=True,
        verbose=True,
        deterministic=True,
        single_cls=False,
        save=True,
        freeze=None,
        augment=True
    )
