import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import math
import os
import time

# === Utility functions ===
def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba * mag_bc == 0:
        return 0
    angle = math.acos(max(min(dot / (mag_ba * mag_bc), 1), -1))
    return math.degrees(angle)

def wrist_velocity(x_hist, y_hist):
    if len(x_hist) < 2:
        return 0
    dx = x_hist[-1] - x_hist[-2]
    dy = y_hist[-1] - y_hist[-2]
    return (dx**2 + dy**2)**0.5

st.title("🏉 Golf Swing Analyzer")
st.write("Upload a video of your golf swing (MP4 format). The app will analyze key phases and give feedback.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Video preview using st.video
    st.subheader("🎥 Uploaded Video Preview")
    st.video(video_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    wrist_x_hist, wrist_y_hist = [], []
    phase = "preparation"
    backswing_started = False
    top_detected = False
    impact_detected = False
    followthrough_started = False
    feedback_summary = []

    frame_idx = 0
    knee_margin = 0.05
    impact_y_margin = 0.05
    impact_velocity = 0.015

    frames = []

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            rwrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            rknee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            rshoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            relbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            lshoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            lhip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]

            wrist_x, wrist_y = rwrist.x, rwrist.y
            wrist_x_hist.append(wrist_x)
            wrist_y_hist.append(wrist_y)

            if not backswing_started:
                if wrist_y < rhip.y:
                    backswing_started = True
                    phase = "backswing"
                    feedback_summary.append("[BACKSWING] Detected")
            elif backswing_started and not top_detected:
                if len(wrist_y_hist) >= 3 and wrist_y_hist[-2] < wrist_y_hist[-3] and wrist_y_hist[-2] < wrist_y_hist[-1]:
                    top_detected = True
                    phase = "downswing"
                    feedback_summary.append("[DOWNSWING] Detected")
            elif top_detected and not impact_detected:
                vel = wrist_velocity(wrist_x_hist, wrist_y_hist)
                if vel > impact_velocity and abs(wrist_y - rknee.y) < impact_y_margin:
                    impact_detected = True
                    phase = "impact"
                    feedback_summary.append("[IMPACT] Detected")
            elif impact_detected and not followthrough_started:
                if wrist_y < rshoulder.y:
                    followthrough_started = True
                    phase = "followthrough"
                    feedback_summary.append("[FOLLOW-THROUGH] Detected")

            elbow_angle = calculate_angle([rshoulder.x, rshoulder.y], [relbow.x, relbow.y], [rwrist.x, rwrist.y])
            shoulder_angle = calculate_angle([lhip.x, lhip.y], [lshoulder.x, lshoulder.y], [rshoulder.x, rshoulder.y])
            hip_angle = calculate_angle([lhip.x, lhip.y], [rhip.x, rhip.y], [rshoulder.x, rshoulder.y])

            if phase == "backswing" and "[BACKSWING FEEDBACK]" not in feedback_summary:
                if shoulder_angle < 40 or elbow_angle < 70:
                    feedback_summary.append("[BACKSWING FEEDBACK] Limited shoulder turn or bent elbow.")
                else:
                    feedback_summary.append("[BACKSWING FEEDBACK] Good backswing structure.")
            elif phase == "downswing" and "[DOWNSWING FEEDBACK]" not in feedback_summary:
                feedback_summary.append("[DOWNSWING FEEDBACK] Hips could rotate more." if hip_angle < 20 else "[DOWNSWING FEEDBACK] Good hip action.")
            elif phase == "impact" and "[IMPACT FEEDBACK]" not in feedback_summary:
                feedback_summary.append("[IMPACT FEEDBACK] Maintain posture through impact zone.")
            elif phase == "followthrough" and "[FOLLOWTHROUGH FEEDBACK]" not in feedback_summary:
                feedback_summary.append("[FOLLOWTHROUGH FEEDBACK] Let the club release fully.")

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Phase: {phase}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frames.append(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        frame_idx += 1

    cap.release()
    try:
        os.remove(video_path)
    except PermissionError:
        st.warning("Temporary video file could not be deleted (maybe still in use).")

    st.subheader("📜 Feedback Summary")
    if feedback_summary:
        unique_feedback = []
        for line in feedback_summary:
            if line not in unique_feedback:
                unique_feedback.append(line)
                st.write(line)

        feedback_file = "swing_feedback.txt"
        with open(feedback_file, "w") as f:
            for line in unique_feedback:
                f.write(line + "\n")

        with open(feedback_file, "rb") as f:
            st.download_button("📄 Download Feedback", f, file_name=feedback_file)
    else:
        st.write("No feedback could be generated.")

    st.subheader("▶️ Replay your swing")
    speed = st.slider("Adjust playback speed (frames per second)", min_value=1, max_value=60, value=30)
    img_placeholder = st.empty()
    for frame in frames:
        img_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(1 / speed)

else:
    st.info("Please upload a video file to analyze your golf swing.")
