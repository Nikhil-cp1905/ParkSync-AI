# Importing necessary libraries
import cv2
import numpy as np 
import pickle
import datetime
import json
from collections import defaultdict

class ParkingLotAnalyzer:
    def __init__(self, pos_list_file='car_park_pos', width=27, height=15):
        # Loading parking space coordinates
        with open(pos_list_file, 'rb') as f:
            self.pos_list = pickle.load(f)
        self.width = width
        self.height = height
        self.parking_history = defaultdict(list)  # Tracks space occupancy over time
        self.start_time = datetime.datetime.now()
        self.occupancy_threshold = 110  # Threshold for determining if space is occupied
        
    def process_frame(self, frame):
        """Process a single frame and return analyzed data"""
        # Image processing pipeline
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (3,3), 1)
        threshold_frame = cv2.adaptiveThreshold(blurred_frame, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 25, 16)
        frame_median = cv2.medianBlur(threshold_frame, 5)
        kernel = np.ones((5, 5), np.uint8)
        dilated_frame = cv2.dilate(frame_median, kernel, iterations=1)
        
        return self.analyze_parking_spaces(frame, dilated_frame)
    
    def analyze_parking_spaces(self, original_frame, processed_frame):
        """Analyze parking spaces and return occupancy data"""
        free_spaces = 0
        space_data = []
        timestamp = datetime.datetime.now()
        
        for idx, pos in enumerate(self.pos_list):
            # Analyze individual parking space
            img_crop = processed_frame[pos[1]:pos[1] + self.height, 
                                    pos[0]:pos[0] + self.width]
            pixel_count = cv2.countNonZero(img_crop)
            is_occupied = pixel_count > self.occupancy_threshold
            
            # Record occupancy history
            self.parking_history[idx].append({
                'timestamp': timestamp,
                'occupied': is_occupied,
                'confidence': pixel_count
            })
            
            # Update visualization
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            if not is_occupied:
                free_spaces += 1
                
            # Draw space identifier and status
            cv2.rectangle(original_frame, pos, 
                         (pos[0] + self.width, pos[1] + self.height), 
                         color, 2)
            cv2.putText(original_frame, f'#{idx}', 
                       (pos[0], pos[1] + self.height - 5),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            
            space_data.append({
                'space_id': idx,
                'occupied': is_occupied,
                'confidence': pixel_count
            })
        
        # Add overall statistics
        self.draw_statistics(original_frame, free_spaces)
        return original_frame, space_data
    
    def draw_statistics(self, frame, free_spaces):
        """Draw statistical information on the frame"""
        total_spaces = len(self.pos_list)
        occupancy_rate = ((total_spaces - free_spaces) / total_spaces) * 100
        
        # Draw overall statistics
        cv2.putText(frame, f'Free: {free_spaces}/{total_spaces}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f'Occupancy: {occupancy_rate:.1f}%', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    def save_analytics(self, filename='parking_analytics.json'):
        """Save parking analytics to a JSON file"""
        analytics = {
            'session_start': self.start_time.isoformat(),
            'session_end': datetime.datetime.now().isoformat(),
            'total_spaces': len(self.pos_list),
            'space_history': {str(k): v for k, v in self.parking_history.items()}
        }
        with open(filename, 'w') as f:
            json.dump(analytics, f, indent=2)

def main():
    # Initialize video capture and analyzer
    cap = cv2.VideoCapture("busy_parking_lot.mp4")
    analyzer = ParkingLotAnalyzer()
    
    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_enhanced.mp4', fourcc, 30, 
                         (frame_width, frame_height))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame and get analysis
            processed_frame, space_data = analyzer.process_frame(frame)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display the frame
            cv2.imshow('Parking Lot Analysis', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Cleanup and save analytics
        analyzer.save_analytics()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
