from flask import Flask, jsonify, request
import cv2
import torch
import numpy as np
from PIL import Image
import time
import threading
import requests

app = Flask(__name__)

# Global variable to store the latest command
current_command = {
    "led": 0, 
    "animal_count": 0,
    "animals": [],
    "detection_time": None
}

class AnimalDetector:
    def __init__(self, model_path=None, conf_threshold=0.5):
        """
        Initialize the animal detector
        """
        self.conf_threshold = conf_threshold
        
        # Load YOLOv5 model
        if model_path:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            # Use pretrained YOLOv5 model (contains animal classes)
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Animal classes in COCO dataset
        self.animal_classes = {
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
            'bear', 'zebra', 'giraffe', 'pig'
        }
        
        print(f"Using device: {self.device}")
        print("Animal detector initialized successfully!")
    
    def detect_animals(self, frame):
        """
        Detect animals in the frame
        Returns: frame with detections, animal_present (bool), detection_info, animal_count
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = self.model(rgb_frame)
        
        # Process results
        animal_present = False
        detection_info = []
        animal_count = 0
        detected_animals = []
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            class_name = self.model.names[int(class_id)]
            
            # Check if detected object is an animal
            if class_name in self.animal_classes and conf >= self.conf_threshold:
                animal_present = True
                animal_count += 1
                detected_animals.append(class_name)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add labelww
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detection_info.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return frame, animal_present, detection_info, animal_count, detected_animals

# Initialize detector
detector = AnimalDetector(conf_threshold=0.5)

def run_animal_detection():
    """
    Function to run real-time animal detection in a separate thread
    """
    global current_command
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting real-time animal detection...")
    print("Animal detection is running in background...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect animals
        processed_frame, animal_present, detections, animal_count, detected_animals = detector.detect_animals(frame.copy())
        
        # Update command based on animal detection
        if animal_present:
            current_command["led"] = 1
            current_command["animal_count"] = animal_count
            current_command["animals"] = detected_animals
            current_command["detection_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"Animal detected! Count: {animal_count} - Types: {detected_animals}")
        else:
            current_command["led"] = 0
            current_command["animal_count"] = 0
            current_command["animals"] = []
            current_command["detection_time"] = None
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
        else:
            fps = 0
        
        # Display animal count and status on frame
        status_text = f"Animals: {animal_count} PRESENT" if animal_present else "Animals: 0 NOT PRESENT"
        status_color = (0, 255, 0) if animal_present else (0, 0, 255)
        
        # Draw status information on frame
        cv2.putText(processed_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Display FPS
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display detection details
        cv2.putText(processed_frame, f"Detections: {len(detections)}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display individual animal types if present
        if animal_present:
            animal_types = {}
            for detection in detections:
                animal_type = detection['class']
                animal_types[animal_type] = animal_types.get(animal_type, 0) + 1
            
            y_offset = 130
            for animal_type, count in animal_types.items():
                type_text = f"{animal_type}: {count}"
                cv2.putText(processed_frame, type_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
        
        # Display frame
        cv2.imshow('Animal Detection', processed_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

@app.route('/get_command', methods=['GET'])
def get_command():
    """
    Endpoint for ESP32 to get the current command
    """
    return jsonify(current_command)

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint to check if animal detection is running
    """
    return jsonify({
        "status": "running",
        "current_command": current_command,
        "message": "Animal detection system is active"
    })

@app.route('/detection_info', methods=['GET'])
def detection_info():
    """
    Endpoint to get detailed detection information
    """
    return jsonify({
        "animal_detected": current_command["led"] == 1,
        "animal_count": current_command["animal_count"],
        "animals": current_command["animals"],
        "detection_time": current_command["detection_time"],
        "timestamp": time.time()
    })

@app.route('/manual_control', methods=['POST'])
def manual_control():
    """
    Endpoint for manual control of LED
    """
    global current_command
    data = request.get_json()
    
    if data and 'led' in data:
        led_value = data['led']
        if led_value in [0, 1]:
            current_command["led"] = led_value
            if led_value == 0:
                current_command["animal_count"] = 0
                current_command["animals"] = []
                current_command["detection_time"] = None
            return jsonify({
                "status": "success", 
                "message": f"LED set to {led_value}",
                "current_command": current_command
            })
    
    return jsonify({"status": "error", "message": "Invalid request"}), 400

if __name__ == '__main__':
    # Start animal detection in a separate thread
    detection_thread = threading.Thread(target=run_animal_detection, daemon=True)
    detection_thread.start()
    
    # Run Flask app

    app.run(host='0.0.0.0', port=5000, debug=False)