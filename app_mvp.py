import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to analyze video
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    pose_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            pose_frames += 1

    cap.release()
    return frame_count, pose_frames

# Streamlit UI
st.set_page_config(page_title="Golf Swing Analyzer", layout="centered")
st.title("🏌️ Golf Swing Analyzer")

uploaded_file = st.file_uploader("Upload a golf swing video", type=["mp4", "mov", "avi"])
if uploaded_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Display video in Streamlit
    st.video(uploaded_file)

    # Analyze video
    with st.spinner("Analyzing golf swing..."):
        total_frames, pose_frames = analyze_video(tmp_path)

    st.success(f"✅ Analysis complete!")
    st.markdown(f"- Total frames: **{total_frames}**")
    st.markdown(f"- Frames with detected pose landmarks: **{pose_frames}**")

    # Cleanup
    os.remove(tmp_path)
