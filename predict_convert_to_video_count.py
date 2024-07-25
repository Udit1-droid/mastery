import cv2
from ultralytics import YOLO
import os
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template
from threading import Thread

# Load the trained YOLOv8 model
model = YOLO(r"E:\chilika_vd\dataset\training_run_yolov8ss\weights\best.pt")

# Paths to the video and output directory
video_path = r"E:\chilika_vd\ch01_20240618122238_3_0_F4AA.mp4"
output_dir = r"E:\chilika_vd\output"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Overall class frequency counter
total_class_counts = defaultdict(int)

# Initialize DeepSort tracker
deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Generate a unique color for each class ID
def generate_color():
    return [random.randint(0, 255) for _ in range(3)]

class_colors = {i: generate_color() for i in range(80)}  # Adjust the range based on the number of classes

# Function to process a single video file
def process_video(video_file):
    vid = cv2.VideoCapture(video_file)
    
    # Get video properties
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    output_video_path = os.path.join(output_dir, f'annotated_{os.path.basename(video_file)}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    class_counts = defaultdict(int)
    
    while True:
        success, frame = vid.read()
        if not success:
            break
        
        # Make predictions
        results = model(frame)
        
        # Initialize list to store detections for DeepSort
        detections = []
        
        # Extract detections from YOLO results
        for result in results[0].boxes:
            if len(result.xyxy[0]) >= 6:
                x1, y1, x2, y2, conf, class_id = result.xyxy[0].cpu().numpy()
                class_id = int(class_id)
                class_counts[class_id] += 1
                total_class_counts[class_id] += 1  # Update total class counts
                
                # Filter out low confidence detections
                if conf > 0.5:  # Adjust the confidence threshold as needed
                    detections.append([x1, y1, x2, y2, conf])
        
        # Update DeepSort tracker with detections and get updated tracks
        tracks = deepsort.update_tracks(detections, frame=frame)  # Pass the frame if needed for feature extraction

        # Draw tracked objects on the frame
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr().astype(int)
            track_id = track.track_id
            class_id = track.class_id  # Assuming track has a class_id attribute
            color = class_colors.get(class_id, (255, 0, 0))  # Default to red if class_id not found
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Track {int(track_id)} - Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Write the frame into the output video
        out.write(frame)
        frame_count += 1

        # Display the frame (optional)
        cv2.imshow('Annotated Frame', frame)
        
        # Break on 'q' key press (for real-time viewing)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()
    
    return class_counts

# Flask application setup
#app = Flask(__name__)

# Function to get and format class counts
def get_class_counts():
    class_counts = defaultdict(int)
    for video_file in os.listdir(video_path):
        video_file_path = os.path.join(video_path, video_file)
        counts = process_video(video_file_path)
        for class_id, count in counts.items():
            class_counts[class_id] += count
    return class_counts

# Route for dashboard
#@app.route('/')
#def dashboard():
    # Get class counts
#    class_counts = get_class_counts()
    
    # Prepare data for table
#    table_data = [{'Class ID': class_id, 'Count': count} for class_id, count in class_counts.items()]
    
    # Prepare data for bar chart
#    class_ids = list(class_counts.keys())
#    counts = list(class_counts.values())
#    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(class_ids))]  # Generate random colors
    
#    plt.figure(figsize=(10, 6))
#    plt.bar(class_ids, counts, color=colors)
#    plt.xlabel('Class ID')
#    plt.ylabel('Count')
#    plt.title('Class Distribution')
#    plt.savefig('static/bar_chart.png')  # Save bar chart image
    
#    return render_template('dashboard.html', table_data=table_data)

#if __name__ == '__main__':
    # Start Flask app in a separate thread
#    Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()