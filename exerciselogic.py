import numpy as np
from datetime import datetime
from collections import Counter


class ExerciseTracker:
    def __init__(self):
        self.reset_session()
        self.exercise_detection_buffer = []
        self.detection_buffer_size = 5

    def reset_session(self):
        """Reset all tracking variables"""
        self.squat_counter = 0
        self.shoulder_press_counter = 0
        self.plank_counter = 0

        self.squat_stage = "up"
        self.shoulder_press_stage = "down"
        self.plank_active = False

        self.current_exercise = None
        self.feedback_message = "Stand in view"
        self.form_score = 0

        self.rep_history = []
        self.exercise_detection_buffer = []

    # ------------------------------------------------------------------
    # EXERCISE DETECTION
    # ------------------------------------------------------------------
    def detect_exercise(self, angles, coords):
        """Auto-detect exercise using simple heuristic rules"""

        avg_elbow = (angles['left_elbow'] + angles['right_elbow']) / 2
        avg_knee = (angles['left_knee'] + angles['right_knee']) / 2
        avg_hip = (angles['left_hip'] + angles['right_hip']) / 2

        detected_exercise = None

        # -------------------- Squat Detection --------------------
        knee_diff = abs(angles['left_knee'] - angles['right_knee'])

        shoulder_y = (coords['left_shoulder'][1] + coords['right_shoulder'][1]) / 2
        hip_y = (coords['left_hip'][1] + coords['right_hip'][1]) / 2

        is_upright = abs(shoulder_y - hip_y) < 200
        knees_similar = knee_diff < 25
        knees_bending = avg_knee < 170
        upright_posture = avg_hip > 130
        feet_visible = coords['left_ankle'][2] > 0.5 and coords['right_ankle'][2] > 0.5

        if is_upright and knees_similar and knees_bending and upright_posture and feet_visible:
            detected_exercise = "squat"

        # -------------------- Shoulder Press Detection --------------------
        if detected_exercise is None:
            hands_above_shoulders = (
                coords['left_wrist'][1] < coords['left_shoulder'][1] and
                coords['right_wrist'][1] < coords['right_shoulder'][1]
            )

            if avg_elbow > 150 and hands_above_shoulders:
                detected_exercise = "shoulder_press"

        # -------------------- Plank Detection --------------------
        if detected_exercise is None:
            ankle_y = (coords['left_ankle'][1] + coords['right_ankle'][1]) / 2
            body_straight = abs((shoulder_y - hip_y) - (hip_y - ankle_y)) < 40
            elbows_bent = avg_elbow < 120

            if body_straight and elbows_bent:
                detected_exercise = "plank"

        # -------------------- Detection Buffer --------------------
        self.exercise_detection_buffer.append(detected_exercise)
        if len(self.exercise_detection_buffer) > self.detection_buffer_size:
            self.exercise_detection_buffer.pop(0)

        valid = [e for e in self.exercise_detection_buffer if e is not None]
        if valid:
            most_common = Counter(valid).most_common(1)
            if most_common[0][1] >= 3:
                return most_common[0][0]

        return None

    # ------------------------------------------------------------------
    # SQUAT TRACKING (UNCHANGED)
    # ------------------------------------------------------------------
    def track_squat(self, angles, coords):
        avg_knee = (angles['left_knee'] + angles['right_knee']) / 2
        avg_hip = (angles['left_hip'] + angles['right_hip']) / 2

        if avg_knee > 160 and self.squat_stage == "down":
            self.squat_counter += 1
            self.squat_stage = "up"
            self.rep_history.append({
                'exercise': 'squat',
                'rep': self.squat_counter,
                'score': self.form_score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        if avg_knee < 110 and self.squat_stage == "up":
            self.squat_stage = "down"

        score = 100
        feedback = []

        if self.squat_stage == "down" and avg_knee > 130:
            score -= 20
            feedback.append("Go deeper")

        if abs(angles['left_knee'] - angles['right_knee']) > 20:
            score -= 15
            feedback.append("Keep knees aligned")

        if avg_hip < 130:
            score -= 20
            feedback.append("Keep back straight")

        self.form_score = max(0, score)
        self.feedback_message = " | ".join(feedback) if feedback else "Good form!"

        return self.squat_counter, self.form_score, self.feedback_message

    # ------------------------------------------------------------------
    # SHOULDER PRESS TRACKING
    # ------------------------------------------------------------------
    def track_shoulder_press(self, angles, coords):
        avg_elbow = (angles['left_elbow'] + angles['right_elbow']) / 2

        if avg_elbow > 160 and self.shoulder_press_stage == "up":
            self.shoulder_press_counter += 1
            self.shoulder_press_stage = "down"
            self.rep_history.append({
                'exercise': 'shoulder_press',
                'rep': self.shoulder_press_counter,
                'score': self.form_score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        if avg_elbow < 90:
            self.shoulder_press_stage = "up"

        score = 100
        if avg_elbow < 80:
            score -= 20

        self.form_score = max(0, score)
        self.feedback_message = "Press arms fully overhead"

        return self.shoulder_press_counter, self.form_score, self.feedback_message

    # ------------------------------------------------------------------
    # PLANK TRACKING (STATIC POSTURE)
    # ------------------------------------------------------------------
    def track_plank(self, angles, coords):
        shoulder_y = (coords['left_shoulder'][1] + coords['right_shoulder'][1]) / 2
        hip_y = (coords['left_hip'][1] + coords['right_hip'][1]) / 2
        ankle_y = (coords['left_ankle'][1] + coords['right_ankle'][1]) / 2

        alignment_error = abs((shoulder_y - hip_y) - (hip_y - ankle_y))

        score = 100
        if alignment_error > 50:
            score -= 30
            feedback = "Keep body straight"
        else:
            feedback = "Good plank posture"

        self.form_score = max(0, score)
        self.feedback_message = feedback

        return self.plank_counter, self.form_score, self.feedback_message

    # ------------------------------------------------------------------
    # FRAME PROCESSING
    # ------------------------------------------------------------------
    def process_frame(self, angles, coords, manual_exercise=None):
        if manual_exercise:
            self.current_exercise = manual_exercise
        else:
            detected = self.detect_exercise(angles, coords)
            if detected:
                self.current_exercise = detected

        if self.current_exercise == "squat":
            return self.track_squat(angles, coords)
        elif self.current_exercise == "shoulder_press":
            return self.track_shoulder_press(angles, coords)
        elif self.current_exercise == "plank":
            return self.track_plank(angles, coords)
        else:
            self.feedback_message = "No exercise detected"
            return 0, 0, self.feedback_message

    # ------------------------------------------------------------------
    # SESSION STATS
    # ------------------------------------------------------------------
    def get_total_reps(self):
        return self.squat_counter + self.shoulder_press_counter + self.plank_counter

    def get_session_stats(self):
        return {
            'squats': self.squat_counter,
            'shoulder_press': self.shoulder_press_counter,
            'plank': self.plank_counter,
            'total': self.get_total_reps(),
            'avg_score': np.mean([r['score'] for r in self.rep_history]) if self.rep_history else 0
        }
