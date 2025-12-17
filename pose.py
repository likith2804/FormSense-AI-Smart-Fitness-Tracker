import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Smoothing buffer for angles
        self.angle_buffer = {
            'left_knee': deque(maxlen=5),
            'right_knee': deque(maxlen=5),
            'left_hip': deque(maxlen=5),
            'right_hip': deque(maxlen=5),
            'left_elbow': deque(maxlen=5),
            'right_elbow': deque(maxlen=5),
            'left_shoulder': deque(maxlen=5),
            'right_shoulder': deque(maxlen=5),
            'back_angle': deque(maxlen=5),
        }
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def smooth_angle(self, angle_name, angle_value):
        """Smooth angle values using moving average"""
        self.angle_buffer[angle_name].append(angle_value)
        return np.mean(self.angle_buffer[angle_name])
    
    def get_landmarks(self, frame):
        """Process frame and extract pose landmarks"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None, frame
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        return results.pose_landmarks, frame
    
    def extract_coordinates(self, landmarks, frame_shape):
        """Extract relevant joint coordinates"""
        h, w, _ = frame_shape
        
        coords = {}
        landmark_list = landmarks.landmark
        
        # Define landmark indices
        landmarks_map = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP.value,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        }
        
        for name, idx in landmarks_map.items():
            lm = landmark_list[idx]
            coords[name] = [lm.x * w, lm.y * h, lm.visibility]
        
        return coords
    
    def calculate_exercise_angles(self, coords):
        """Calculate all relevant angles for exercises"""
        angles = {}
        
        # Knee angles (for squats and lower-body movements)
        left_knee_angle = self.calculate_angle(
            coords['left_hip'][:2],
            coords['left_knee'][:2],
            coords['left_ankle'][:2]
        )
        right_knee_angle = self.calculate_angle(
            coords['right_hip'][:2],
            coords['right_knee'][:2],
            coords['right_ankle'][:2]
        )
        
        angles['left_knee'] = self.smooth_angle('left_knee', left_knee_angle)
        angles['right_knee'] = self.smooth_angle('right_knee', right_knee_angle)
        
        # Hip angles (for squat posture analysis)
        left_hip_angle = self.calculate_angle(
            coords['left_shoulder'][:2],
            coords['left_hip'][:2],
            coords['left_knee'][:2]
        )
        right_hip_angle = self.calculate_angle(
            coords['right_shoulder'][:2],
            coords['right_hip'][:2],
            coords['right_knee'][:2]
        )
        
        angles['left_hip'] = self.smooth_angle('left_hip', left_hip_angle)
        angles['right_hip'] = self.smooth_angle('right_hip', right_hip_angle)
        
        # Elbow angles (upper-body exercises)
        left_elbow_angle = self.calculate_angle(
            coords['left_shoulder'][:2],
            coords['left_elbow'][:2],
            coords['left_wrist'][:2]
        )
        right_elbow_angle = self.calculate_angle(
            coords['right_shoulder'][:2],
            coords['right_elbow'][:2],
            coords['right_wrist'][:2]
        )
        
        angles['left_elbow'] = self.smooth_angle('left_elbow', left_elbow_angle)
        angles['right_elbow'] = self.smooth_angle('right_elbow', right_elbow_angle)
        
        # Back angle (shoulder–hip–ankle alignment)
        back_angle_left = self.calculate_angle(
            coords['left_shoulder'][:2],
            coords['left_hip'][:2],
            coords['left_ankle'][:2]
        )
        angles['back_angle'] = self.smooth_angle('back_angle', back_angle_left)
        
        return angles
    
    def release(self):
        """Release resources"""
        self.pose.close()
