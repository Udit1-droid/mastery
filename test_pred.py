import cv2
from ultralytics import YOLO
import os
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort  # Import DeepSort tracker

# Load the trained model using last.pt
model = YOLO(r"E:\chilika_vd\dataset\training_run_yolov8sp\weights\best.pt")

# Path to the video or directory containing video files
video_path = r"E:\chilika_vd\ch01_20240618122238_3_0_F4AA.mp4"
output_dir = r"E:\chilika_vd\output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Overall class frequency counter
total_class_counts = defaultdict(int)

# Initialize DeepSort tracker
deepsort = DeepSort(r'E:\chilika_vd\Required_codes/ckpt.t7')  # Provide path to the DeepSort model checkpoint

# Function to process a single video file
def process_video(video_file):
    vid = cv2.VideoCapture(video_file)
    
    # Get video properties
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    output_video_path = os.path.join(output_dir, f'annotated_{os.path.basename(video_file)}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    # Class frequency counter for the current video
    class_counts = defaultdict(int)
    
    while True:
        success, frame = vid.read()
        if not success:
            break
        
        # Make predictions
        results = model(frame)
        
        # Initialize lists to store detections and features for DeepSort
        detections = []
        features = []
        
        # Extract detections from YOLO results
        if results and results[0].boxes:  # Check if there are any detections
            for result in results[0].boxes:
                if len(result.xyxy[0]) >= 6:  # Ensure enough values for unpacking
                    x1, y1, x2, y2, conf, class_id = result.xyxy[0].cpu().numpy()
                    class_id = int(class_id)
                    class_counts[class_id] += 1
                    
                    # Store detection in DeepSort format (left, top, right, bottom, confidence)
                    detections.append([x1, y1, x2, y2, conf])
                    
                    # Extract features if needed for DeepSort (e.g., from a feature extractor network)
                    # features.append(...)  # Extract and append features if available
        
        # Update DeepSort tracker with detections and get updated tracks
        tracks = deepsort.update(detections)

        # Draw tracked objects on the frame
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Track {int(track_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Write the frame into the output video
        out.write(frame)
        frame_count += 1

        # Display the frame (optional)
        cv2.imshow('Annotated Frame', frame)
        
        # Break on 'q' key press (for real-time viewing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    
    # Aggregate counts for the current video into the total counts
    for class_id, count in class_counts.items():
        total_class_counts[class_id] += count

    cv2.destroyAllWindows()

# Check if the input path is a file or a directory
if os.path.isfile(video_path):
    process_video(video_path)
elif os.path.isdir(video_path):
    for video_file in os.listdir(video_path):
        process_video(os.path.join(video_path, video_file))
else:
    print(f"Invalid path: {video_path}")

# Print total class frequencies
for class_id, count in total_class_counts.items():
    class_name = model.names[class_id]
    print(f"{class_name}: {count}")
