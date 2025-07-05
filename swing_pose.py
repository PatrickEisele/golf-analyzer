import cv2
import mediapipe as mp
import math

def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba * mag_bc == 0:
        return 0

    cosine_angle = dot_product / (mag_ba * mag_bc)
    angle = math.acos(max(min(cosine_angle, 1), -1))
    return math.degrees(angle)

def detect_impact(wrist_x_hist, threshold=0.05):
    if len(wrist_x_hist) < 5:
        return False
    vels = [abs(wrist_x_hist[i+1] - wrist_x_hist[i]) for i in range(len(wrist_x_hist)-1)]
    if vels[-1] < threshold and max(vels[-5:-1]) > threshold:
        return True
    return False

def wrist_velocity(wrist_x_hist, wrist_y_hist):
    if len(wrist_x_hist) < 2 or len(wrist_y_hist) < 2:
        return 0
    dx = wrist_x_hist[-1] - wrist_x_hist[-2]
    dy = wrist_y_hist[-1] - wrist_y_hist[-2]
    return (dx**2 + dy**2)**0.5

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("swing.mp4")

wrist_y_history = []
wrist_x_history = []

backswing_top_frame = None
impact_frame = None
frame_idx = 0

prep_velocity_threshold = 0.003  # adjust as needed
prep_frames_required = 20        # increased to make it stricter
prep_frame_count = 0
min_upward_movement = 0.03       # minimum wrist y upward movement after prep

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        wrist_y_history.append(right_wrist_y)
        wrist_x_history.append(right_wrist_x)

        vel = wrist_velocity(wrist_x_history, wrist_y_history)

        # Preparation phase detection: wrist velocity low for several frames
        if backswing_top_frame is None and prep_frame_count < prep_frames_required:
            if vel < prep_velocity_threshold:
                prep_frame_count += 1
            else:
                prep_frame_count = 0

        # Detect backswing top only if after sufficient preparation AND wrist moved upward enough
        if len(wrist_y_history) >= 5 and backswing_top_frame is None and prep_frame_count >= prep_frames_required:
            prep_end_y = wrist_y_history[-prep_frames_required]
            recent_min_y = min(wrist_y_history[-5:])
            upward_movement = prep_end_y - recent_min_y  # positive means wrist moved upward

            if upward_movement > min_upward_movement:
                # Check local min in wrist y for backswing top
                if (wrist_y_history[-2] < wrist_y_history[-3] and
                    wrist_y_history[-2] < wrist_y_history[-1]):
                    backswing_top_frame = frame_idx - 1
                    print(f"Backswing top detected at frame {backswing_top_frame}")

        # Detect impact: sudden slowdown in wrist X velocity
        if backswing_top_frame is not None and impact_frame is None:
            if detect_impact(wrist_x_history[-10:]):
                impact_frame = frame_idx
                print(f"Impact detected at frame {impact_frame}")

        # Determine current phase
        if prep_frame_count < prep_frames_required:
            phase = "preparation"
        elif backswing_top_frame is None:
            phase = "backswing"
        elif impact_frame is None:
            phase = "downswing"
        else:
            phase = "followthrough"

        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = calculate_angle(left_hip, left_shoulder, right_shoulder)
        hip_angle = calculate_angle(left_hip, right_hip, right_shoulder)

        feedback = []

        # Phase-specific feedback
        if phase == "backswing":
            if left_shoulder_angle < 40:
                feedback.append("Limited shoulder rotation in backswing.")
            if right_elbow_angle < 70:
                feedback.append("Right elbow is too bent in backswing.")
        elif phase == "downswing":
            if hip_angle < 20:
                feedback.append("Hip rotation could be improved in downswing.")
        elif phase == "followthrough":
            pass  # add followthrough feedback if desired

        for f in feedback:
            print(f"[{phase.upper()}] {f}")

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Phase: {phase}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Golf Swing Analysis", frame)
    frame_idx += 1

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
