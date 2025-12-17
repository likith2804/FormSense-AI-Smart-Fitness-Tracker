FormSense AI – Smart Fitness Tracker
FormSense AI is a computer vision based fitness tracking application that analyzes workout posture
and provides real-time feedback using a webcam or video input.
Project Overview
The main objective of this project is to help users perform exercises safely by analyzing posture and
giving instant feedback for correction.
Supported Exercises
• Squats
• Shoulder Press
Technology Stack
• Python
• MediaPipe – Pose estimation
• OpenCV – Video processing
• Streamlit – Web interface
• NumPy & Pandas – Calculations and data handling
Project Files
• app.py – Main Streamlit application
• pose.py – Pose detection and angle calculation
• exerciselogic.py – Exercise rules and feedback logic
• requirements.txt – Project dependencies
How It Works
• User starts webcam or uploads a video.
• Pose landmarks are detected using MediaPipe.
• Joint angles are calculated in real time.
• Exercise rules are applied.
• Feedback is shown on the screen.
Conclusion
FormSense AI demonstrates how AI and computer vision can be applied to fitness applications to
improve exercise form and reduce the risk of injury.
