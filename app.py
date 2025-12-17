import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import os
import time

from pose import PoseDetector
from exerciselogic import ExerciseTracker

# Page config
st.set_page_config(
    page_title="FormSense AI – Smart Fitness Tracker",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.app-header {
    background: linear-gradient(135deg, #1f2933, #111827);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.app-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #22d3ee, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.app-subtitle {
    color: #9ca3af;
    margin-top: 0.5rem;
}

.sidebar-card {
    background: #111827;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.footer-card {
    text-align: center;
    padding: 1.5rem;
    border-radius: 14px;
    background: linear-gradient(90deg, #22d3ee, #6366f1);
    color: black;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if 'exercise_tracker' not in st.session_state:
        st.session_state.exercise_tracker = ExerciseTracker()
    if 'pose_detector' not in st.session_state:
        st.session_state.pose_detector = PoseDetector()
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'recorded_frames' not in st.session_state:
        st.session_state.recorded_frames = []

init_session_state()

def process_uploaded_video(video_file):
    """Process uploaded video file"""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    stats_placeholder = st.empty()
    
    pose_detector = PoseDetector()
    exercise_tracker = ExerciseTracker()
    
    manual_exercise = st.session_state.get('video_exercise_mode', None)
    if manual_exercise == "Auto-detect":
        manual_exercise = None
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        landmarks, frame = pose_detector.get_landmarks(frame)
        
        if landmarks:
            coords = pose_detector.extract_coordinates(landmarks, frame.shape)
            angles = pose_detector.calculate_exercise_angles(coords)
            
            reps, score, feedback = exercise_tracker.process_frame(
                angles, coords, manual_exercise
            )
            
            color = (0, 255, 0) if score >= 70 else (0, 165, 255) if score >= 50 else (0, 0, 255)
            
            cv2.putText(frame, f"Exercise: {exercise_tracker.current_exercise or 'None'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Reps: {reps}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Score: {score}/100", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            y_offset = 150
            for line in feedback.split('|'):
                cv2.putText(frame, line.strip(), 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 40
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)
        
        stats = exercise_tracker.get_session_stats()
        stats_placeholder.markdown(f"""
            **Processing Video**
            Frame {frame_count}/{total_frames} •
            Squats: {stats['squats']} •
            Shoulder Press: {stats.get('shoulder_press', 0)} •
            Plank: {stats.get('plank', 0)} •
            Avg Form Score: {stats['avg_score']:.1f}
            """)

    
    cap.release()
    pose_detector.release()
    
    time.sleep(0.5)
    try:
        os.unlink(tfile.name)
    except:
        pass
    
    progress_bar.empty()
    stframe.empty()
    stats_placeholder.empty()
    
    return exercise_tracker

def create_enhanced_charts(rep_history):
    """Create detailed and easy-to-understand charts"""
    df = pd.DataFrame(rep_history)
    
    # Chart 1: Form Score Timeline
    fig1 = go.Figure()
    
    fig1.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.1, line_width=0)
    fig1.add_hrect(y0=70, y1=80, fillcolor="lightgreen", opacity=0.1, line_width=0)
    fig1.add_hrect(y0=50, y1=70, fillcolor="yellow", opacity=0.1, line_width=0)
    fig1.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.1, line_width=0)
    
    for exercise in df['exercise'].unique():
        exercise_data = df[df['exercise'] == exercise].reset_index(drop=True)
        
        colors = []
        for score in exercise_data['score']:
            if score >= 80:
                colors.append('green')
            elif score >= 70:
                colors.append('lightgreen')
            elif score >= 50:
                colors.append('orange')
            else:
                colors.append('red')
        
        fig1.add_trace(go.Scatter(
            x=list(range(1, len(exercise_data) + 1)),
            y=exercise_data['score'],
            mode='lines+markers',
            name=exercise.capitalize(),
            line=dict(width=3),
            marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
            text=[f"Rep {i+1}<br>Score: {s}/100" for i, s in enumerate(exercise_data['score'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    fig1.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent (80+)")
    fig1.add_hline(y=70, line_dash="dash", line_color="lightgreen", annotation_text="Good (70-80)")
    fig1.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Fair (50-70)")
    
    fig1.update_layout(
        title="📊 Form Quality Timeline",
        xaxis_title="Rep Number",
        yaxis_title="Form Score (0-100)",
        yaxis=dict(range=[0, 105]),
        height=500,
        showlegend=True
    )
    
    # Chart 2: Exercise Distribution
    exercise_counts = df['exercise'].value_counts()
    
    fig2 = go.Figure(data=[go.Pie(
        labels=[e.capitalize() for e in exercise_counts.index],
        values=exercise_counts.values,
        hole=0.4,
        textinfo='label+percent+value',
        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
    )])
    
    fig2.update_layout(
        title="💪 Exercise Distribution",
        height=450
    )
    
    # Chart 3: Average Score by Exercise
    avg_scores = df.groupby('exercise')['score'].agg(['mean', 'min', 'max']).reset_index()
    
    fig3 = go.Figure()
    
    colors_bar = []
    for score in avg_scores['mean']:
        if score >= 80:
            colors_bar.append('green')
        elif score >= 70:
            colors_bar.append('lightgreen')
        elif score >= 50:
            colors_bar.append('orange')
        else:
            colors_bar.append('red')
    
    fig3.add_trace(go.Bar(
        x=[e.capitalize() for e in avg_scores['exercise']],
        y=avg_scores['mean'],
        marker_color=colors_bar,
        text=[f'{v:.1f}' for v in avg_scores['mean']],
        textposition='outside'
    ))
    
    fig3.update_layout(
        title="⭐ Average Form Quality",
        xaxis_title="Exercise Type",
        yaxis_title="Average Score",
        yaxis=dict(range=[0, 105]),
        height=450
    )
    
    # Chart 4: Rep Table
    fig4 = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Rep #</b>', '<b>Exercise</b>', '<b>Score</b>', '<b>Quality</b>', '<b>Time</b>'],
            fill_color='#667eea',
            font=dict(color='white', size=14),
            align='center'
        ),
        cells=dict(
            values=[
                df.index + 1,
                [e.capitalize() for e in df['exercise']],
                [f"{s:.0f}/100" for s in df['score']],
                ['🌟 Excellent' if s >= 80 else '✅ Good' if s >= 70 else '⚠️ Fair' if s >= 50 else '❌ Poor' 
                 for s in df['score']],
                df['timestamp']
            ],
            align='center'
        )
    )])
    
    fig4.update_layout(
        title="📋 Detailed Rep Analysis",
        height=400
    )
    
    return fig1, fig2, fig3, fig4

def generate_detailed_feedback(rep_history):
    """Generate comprehensive text feedback"""
    df = pd.DataFrame(rep_history)
    
    feedback = []
    feedback.append("## 📊 PERFORMANCE ANALYSIS\n")
    
    total_reps = len(df)
    avg_score = df['score'].mean()
    best_score = df['score'].max()
    worst_score = df['score'].min()
    
    feedback.append("### 🎯 OVERALL PERFORMANCE\n")
    
    if avg_score >= 80:
        grade = "A (Excellent)"
        emoji = "🌟"
        comment = "Outstanding form! Keep up the excellent work."
    elif avg_score >= 70:
        grade = "B (Good)"
        emoji = "✅"
        comment = "Good performance with room for minor improvements."
    elif avg_score >= 60:
        grade = "C (Fair)"
        emoji = "⚠️"
        comment = "Fair performance. Focus on the feedback to improve."
    else:
        grade = "D (Needs Improvement)"
        emoji = "❌"
        comment = "Keep practicing and focus on proper form."
    
    feedback.append(f"**{emoji} Overall Grade: {grade}**\n")
    feedback.append(f"*{comment}*\n")
    feedback.append(f"- **Total Reps:** {total_reps}")
    feedback.append(f"- **Average Score:** {avg_score:.1f}/100")
    feedback.append(f"- **Best Rep:** {best_score:.0f}/100")
    feedback.append(f"- **Worst Rep:** {worst_score:.0f}/100\n")
    
    excellent = len(df[df['score'] >= 80])
    good = len(df[(df['score'] >= 70) & (df['score'] < 80)])
    fair = len(df[(df['score'] >= 50) & (df['score'] < 70)])
    poor = len(df[df['score'] < 50])
    
    feedback.append("**Score Distribution:**")
    feedback.append(f"- 🌟 Excellent: {excellent} reps ({excellent/total_reps*100:.1f}%)")
    feedback.append(f"- ✅ Good: {good} reps ({good/total_reps*100:.1f}%)")
    feedback.append(f"- ⚠️ Fair: {fair} reps ({fair/total_reps*100:.1f}%)")
    feedback.append(f"- ❌ Poor: {poor} reps ({poor/total_reps*100:.1f}%)\n")
    
    feedback.append("---\n")
    feedback.append("### 💪 EXERCISE BREAKDOWN\n")
    
    for exercise in df['exercise'].unique():
        ex_data = df[df['exercise'] == exercise]
        ex_count = len(ex_data)
        ex_avg = ex_data['score'].mean()
        ex_best = ex_data['score'].max()
        ex_worst = ex_data['score'].min()
        
        if ex_avg >= 80:
            ex_emoji = "🌟"
            ex_grade = "Excellent"
        elif ex_avg >= 70:
            ex_emoji = "✅"
            ex_grade = "Good"
        elif ex_avg >= 50:
            ex_emoji = "⚠️"
            ex_grade = "Fair"
        else:
            ex_emoji = "❌"
            ex_grade = "Needs Work"
        
        feedback.append(f"**{ex_emoji} {exercise.upper()} - {ex_grade}**")
        feedback.append(f"- Reps: {ex_count} | Avg: {ex_avg:.1f}/100 | Best: {ex_best:.0f} | Worst: {ex_worst:.0f}\n")
    
    feedback.append("---\n")
    feedback.append("### 💡 RECOMMENDATIONS\n")
    
    if avg_score >= 80:
        feedback.append("- 🏆 Excellent work! Maintain this quality")
        feedback.append("- Consider increasing intensity or reps")
    elif avg_score >= 70:
        feedback.append("- Focus on consistency across all reps")
        feedback.append("- Review feedback for exercises below 80")
    else:
        feedback.append("- Practice with focus on form, not speed")
        feedback.append("- Watch tutorial videos for proper technique")
        feedback.append("- Consider working with a trainer")
    
    return "\n".join(feedback)

def main():
    # Header
    st.markdown("""
<div class="app-header">
    <div class="app-title">🏋️ FormSense AI</div>
    <div class="app-subtitle">
        Real-time form checking • Rep quality scoring • Posture awareness
    </div>
</div>
""", unsafe_allow_html=True)

    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-card'><h3>Control Panel</h3></div>", unsafe_allow_html=True)

        
        mode = st.radio("📍 Mode", ["📹 Live Webcam", "📁 Upload Video"])
        
        st.markdown("---")
        exercise_mode = st.selectbox(
    "🎯 Exercise Mode",
    ["Auto-detect", "squat", "shoulder_press", "plank"]
)
        st.session_state.video_exercise_mode = exercise_mode
        
        st.markdown("---")
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.exercise_tracker.reset_session()
            st.session_state.recorded_frames = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("<div class='sidebar-card'><h4>📊 Live Stats</h4></div>", unsafe_allow_html=True)

        stats = st.session_state.exercise_tracker.get_session_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🏋️ Squats", stats.get('squats', 0))
        st.metric("🏋️ Shoulder Press", stats.get('shoulder_press', 0))

    with col2:
        st.metric("🧘 Plank Holds", stats.get('plank', 0))
        st.metric("📈 Total", stats.get('total', 0))

    if stats.get('total', 0) > 0:
        st.metric("⭐ Avg Score", f"{stats.get('avg_score', 0):.1f}/100")

    
    # Main content
    if mode == "📹 Live Webcam":
        st.subheader("📹 Live Webcam")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🔴 START RECORDING", use_container_width=True, type="primary", disabled=st.session_state.is_recording):
                st.session_state.is_recording = True
                st.session_state.recorded_frames = []
                st.rerun()
        with col2:
            if st.button("⏹️ STOP RECORDING", use_container_width=True, disabled=not st.session_state.is_recording):
                st.session_state.is_recording = False
                st.rerun()
        with col3:
            if st.session_state.recorded_frames:
                st.success(f"✅ {len(st.session_state.recorded_frames)} frames")
            else:
                st.info("Ready to record")
        with col4:
            if st.button("🗑️ CLEAR", use_container_width=True):
                st.session_state.recorded_frames = []
                st.session_state.is_recording = False
                if 'camera_active' in st.session_state:
                    st.session_state.camera_active = False
                st.rerun()
        
        # Process recorded video FIRST - if frames exist and not recording
        if st.session_state.recorded_frames and not st.session_state.is_recording:
            st.markdown("---")
            st.info(f"📹 Recording ready: {len(st.session_state.recorded_frames)} frames (~{len(st.session_state.recorded_frames)//20} seconds)")
            
            if st.button("🔄 PROCESS RECORDING", type="primary", use_container_width=True):
                with st.spinner("Processing your recording..."):
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    height, width = st.session_state.recorded_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_path, fourcc, 20.0, (width, height))
                    
                    for frame in st.session_state.recorded_frames:
                        out.write(frame)
                    out.release()
                    
                    # Process video
                    cap = cv2.VideoCapture(temp_path)
                    stframe = st.empty()
                    
                    pose_detector = PoseDetector()
                    exercise_tracker = ExerciseTracker()
                    manual_exercise = exercise_mode if exercise_mode != "Auto-detect" else None
                    
                    frame_count = 0
                    total_frames = len(st.session_state.recorded_frames)
                    progress_bar = st.progress(0)
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)
                        
                        landmarks, frame = pose_detector.get_landmarks(frame)
                        
                        if landmarks:
                            coords = pose_detector.extract_coordinates(landmarks, frame.shape)
                            angles = pose_detector.calculate_exercise_angles(coords)
                            reps, score, feedback = exercise_tracker.process_frame(angles, coords, manual_exercise)
                            
                            color = (0, 255, 0) if score >= 70 else (0, 165, 255) if score >= 50 else (0, 0, 255)
                            cv2.putText(frame, f"Exercise: {exercise_tracker.current_exercise or 'None'}", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, f"Reps: {reps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(frame, f"Score: {score}/100", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    cap.release()
                    pose_detector.release()
                    
                    time.sleep(0.5)
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    progress_bar.empty()
                    stframe.empty()
                    
                    st.success("✅ Analysis Complete!")
                    
                    if exercise_tracker.rep_history:
                        st.markdown("---")
                        
                        # Show last frame
                        st.markdown("### 📸 Final Frame")
                        last_frame = st.session_state.recorded_frames[-1]
                        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                        st.image(last_frame_rgb, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Stats
                        # Stats
                        stats = exercise_tracker.get_session_stats()
                        col1, col2, col3, col4 = st.columns(4)

                        col1.metric("Total", stats.get('total', 0))
                        col2.metric("Squats", stats.get('squats', 0))
                        col3.metric("Plank Holds", stats.get('plank', 0))
                        col4.metric("Lunges", stats.get('lunges', 0))

                        
                        st.markdown("---")
                        
                        # Feedback
                        st.markdown(generate_detailed_feedback(exercise_tracker.rep_history))
                        
                        st.markdown("---")
                        
                        # Charts
                        fig1, fig2, fig3, fig4 = create_enhanced_charts(exercise_tracker.rep_history)
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig2, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # Download
                        df = pd.DataFrame(exercise_tracker.rep_history)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=csv,
                            file_name=f"fitform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No exercises detected in recording")
        
        # Show camera ONLY if no recorded frames ready for processing
        elif not st.session_state.recorded_frames or st.session_state.is_recording:
            st.markdown("---")
            camera_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            if 'camera_active' not in st.session_state:
                st.session_state.camera_active = False
            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                camera_toggle = st.checkbox("▶️ Camera ON/OFF", value=st.session_state.camera_active, key="cam_toggle")
                if camera_toggle != st.session_state.camera_active:
                    st.session_state.camera_active = camera_toggle
                    st.rerun()
            
            if st.session_state.camera_active:
                cap = cv2.VideoCapture(0)
                pose_detector = PoseDetector()
                exercise_tracker = ExerciseTracker()
                
                manual_exercise = exercise_mode if exercise_mode != "Auto-detect" else None
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Cannot access webcam")
                        break
                    
                    # Record if active
                    if st.session_state.is_recording:
                        st.session_state.recorded_frames.append(frame.copy())
                    
                    # Process pose
                    landmarks, frame = pose_detector.get_landmarks(frame)
                    
                    if landmarks:
                        coords = pose_detector.extract_coordinates(landmarks, frame.shape)
                        angles = pose_detector.calculate_exercise_angles(coords)
                        
                        reps, score, feedback = exercise_tracker.process_frame(
                            angles, coords, manual_exercise
                        )
                        
                        # Update session state
                        st.session_state.exercise_tracker.squat_counter = exercise_tracker.squat_counter
                        st.session_state.exercise_tracker.current_exercise = exercise_tracker.current_exercise
                        st.session_state.exercise_tracker.feedback_message = feedback
                        st.session_state.exercise_tracker.form_score = score
                        st.session_state.exercise_tracker.rep_history = exercise_tracker.rep_history
                        
                        color = (0, 255, 0) if score >= 70 else (0, 165, 255) if score >= 50 else (0, 0, 255)
                        
                        cv2.putText(frame, f"Exercise: {exercise_tracker.current_exercise or 'None'}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(frame, f"Reps: {reps}", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        cv2.putText(frame, f"Score: {score}/100", 
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
                        # Recording indicator
                        if st.session_state.is_recording:
                            cv2.circle(frame, (frame.shape[1] - 40, 30), 15, (0, 0, 255), -1)
                            cv2.putText(frame, "REC", (frame.shape[1] - 90, 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        y_offset = 150
                        for line in feedback.split('|'):
                            cv2.putText(frame, line.strip(), 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            y_offset += 35
                        
                        feedback_placeholder.info(f"**{exercise_tracker.current_exercise or 'Detecting...'}** | Score: {score}/100")
                    else:
                        cv2.putText(frame, "No pose detected - Stand in frame", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    time.sleep(0.03)
                    
                    # Check if recording stopped
                    if not st.session_state.is_recording and st.session_state.recorded_frames:
                        st.session_state.camera_active = False
                        break
                
                cap.release()
                pose_detector.release()
                
                # If recording stopped with frames, trigger rerun to show process button
                if st.session_state.recorded_frames and not st.session_state.is_recording:
                    st.rerun()
    
    else:  # Upload Video mode
        st.subheader("📁 Upload Video")
        
        uploaded_file = st.file_uploader("Choose video (MP4, MOV, AVI)", type=['mp4', 'mov', 'avi'])
        
        if uploaded_file is not None:
            st.success(f"✅ {uploaded_file.name} ({uploaded_file.size / (1024*1024):.2f} MB)")
            
            if st.button("▶️ ANALYZE VIDEO", type="primary", use_container_width=True):
                with st.spinner("Processing video..."):
                    exercise_tracker = process_uploaded_video(uploaded_file)
                    
                    st.success("✅ Analysis Complete!")
                    
                    if exercise_tracker.rep_history:
                        stats = exercise_tracker.get_session_stats()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total", stats.get('total', 0))
                        col2.metric("Squats", stats.get('squats', 0))
                        col3.metric("Plank Holds", stats.get('plank', 0))
                        col4.metric("Lunges", stats.get('lunges', 0))


                        
                        st.markdown("---")
                        st.markdown(generate_detailed_feedback(exercise_tracker.rep_history))
                        
                        st.markdown("---")
                        fig1, fig2, fig3, fig4 = create_enhanced_charts(exercise_tracker.rep_history)
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig2, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        df = pd.DataFrame(exercise_tracker.rep_history)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=csv,
                            file_name=f"fitform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No exercises detected")
    
    # Footer
    st.markdown("---")
    st.markdown("""
<div class="footer-card">
    <h3>🏋️ FormSense AI</h3>
    <p>AI-powered posture analysis for smarter workouts</p>
    <small>Built with MediaPipe & OpenCV</small>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()