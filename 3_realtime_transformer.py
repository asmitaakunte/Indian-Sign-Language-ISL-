import cv2
import mediapipe as mp
import numpy as np
import pickle
import torch
import torch.nn as nn
import threading
import queue
import time
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "isl_transformer_holistic.pth"
LABEL_PATH = "isl_labels_holistic.pkl"

SEQ_LEN = 30
INPUT_DIM = 260

CONFIDENCE_THRESHOLD = 0.85  # below this = not a known letter (raise it to reject more random gestures)
RANDOM_FRAMES = 15           # hand visible + low confidence this many frames in a row -> "Random gesture"
RANDOM_TEXT = "Random gesture"
STABLE_FRAMES = 10         # same confident letter this many frames in a row -> speak
NO_HAND_RESET_FRAMES = 10  # frames with no hands -> allow the same letter to be spoken again
SPEAK_LABELS = ["A", "B", "C", "D", "E"]

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ISL REALTIME TRANSFORMER")
print("=" * 60)
print("\nDevice:", device)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def angle_between(v1, v2):
    v1 = safe_normalize(v1)
    v2 = safe_normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot) / np.pi


# ============================================================
# HAND FEATURES = 107 (SAME AS TRAINING)
# ============================================================

ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14),
]

FINGER_SEGMENTS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15),
]

FINGERTIPS = [4, 8, 12, 16, 20]

PAIR_LIST = [
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20),
    (16, 20),
]


def extract_hand_features(hand_landmarks):

    if hand_landmarks is None:
        return np.zeros(107, dtype=np.float32)

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    wrist = points[0]

    relative = points - wrist
    scale = np.linalg.norm(relative[9])
    if scale < 1e-6:
        scale = 1.0
    relative = relative / scale
    landmark_features = relative.flatten()

    angle_features = []
    for a, b, c in ANGLE_TRIPLETS:
        v1 = points[a] - points[b]
        v2 = points[c] - points[b]
        angle_features.append(angle_between(v1, v2))
    angle_features = np.array(angle_features, dtype=np.float32)

    direction_features = []
    for a, b in FINGER_SEGMENTS:
        dx = points[b][0] - points[a][0]
        dy = points[b][1] - points[a][1]
        direction_features.append(np.arctan2(dy, dx) / np.pi)
    direction_features = np.array(direction_features, dtype=np.float32)

    tip_wrist_features = np.array(
        [np.linalg.norm(points[tip] - wrist) / scale for tip in FINGERTIPS],
        dtype=np.float32
    )

    pair_features = np.array(
        [np.linalg.norm(points[a] - points[b]) / scale for a, b in PAIR_LIST],
        dtype=np.float32
    )

    v1 = points[5] - points[0]
    v2 = points[17] - points[0]
    palm_features = safe_normalize(np.cross(v1, v2)).astype(np.float32)

    hand_scale = np.array([scale], dtype=np.float32)

    features = np.concatenate([
        landmark_features,
        angle_features,
        direction_features,
        tip_wrist_features,
        pair_features,
        palm_features,
        hand_scale,
    ])

    if len(features) != 107:
        raise ValueError(f"Hand feature count = {len(features)}, expected 107")

    return features.astype(np.float32)


# ============================================================
# POSE FEATURES = 37 (SAME AS TRAINING)
# ============================================================

def extract_pose_features(pose_landmarks):

    if pose_landmarks is None:
        return np.zeros(37, dtype=np.float32)

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark],
        dtype=np.float32
    )

    visibility = np.array(
        [lm.visibility for lm in pose_landmarks.landmark],
        dtype=np.float32
    )

    pose_visibility = visibility[:33]

    left_shoulder = points[11]
    right_shoulder = points[12]
    left_hip = points[23]
    right_hip = points[24]

    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
    hip_distance = np.linalg.norm(left_hip - right_hip)

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0

    torso_distance = np.linalg.norm(shoulder_center - hip_center)

    torso_angle = angle_between(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        hip_center - shoulder_center
    )

    body_features = np.array(
        [shoulder_distance, hip_distance, torso_distance, torso_angle],
        dtype=np.float32
    )

    features = np.concatenate([pose_visibility, body_features])

    if len(features) != 37:
        raise ValueError(f"Pose feature count = {len(features)}, expected 37")

    return features.astype(np.float32)


# ============================================================
# FRAME FEATURES = 260 (SAME AS TRAINING)
# ============================================================

def extract_frame_features(results):

    left_hand = results.left_hand_landmarks
    right_hand = results.right_hand_landmarks

    left_features = extract_hand_features(left_hand)
    right_features = extract_hand_features(right_hand)

    left_present = 1.0 if left_hand is not None else 0.0
    right_present = 1.0 if right_hand is not None else 0.0

    relationship = np.zeros(7, dtype=np.float32)

    if left_hand is not None and right_hand is not None:

        left_points = np.array(
            [[lm.x, lm.y, lm.z] for lm in left_hand.landmark],
            dtype=np.float32
        )
        right_points = np.array(
            [[lm.x, lm.y, lm.z] for lm in right_hand.landmark],
            dtype=np.float32
        )

        wrist_distance = np.linalg.norm(left_points[0] - right_points[0])

        left_palm = np.mean(left_points[[0, 5, 9, 13, 17]], axis=0)
        right_palm = np.mean(right_points[[0, 5, 9, 13, 17]], axis=0)
        palm_distance = np.linalg.norm(left_palm - right_palm)

        fingertip_distances = [
            np.linalg.norm(left_points[i] - right_points[i])
            for i in [4, 8, 12, 16, 20]
        ]

        relationship = np.array(
            [wrist_distance, palm_distance, *fingertip_distances],
            dtype=np.float32
        )

    pose_features = extract_pose_features(results.pose_landmarks)

    frame_features = np.concatenate([
        left_features,
        right_features,
        relationship,
        np.array([left_present, right_present], dtype=np.float32),
        pose_features,
    ])

    if len(frame_features) != 260:
        raise ValueError(
            f"Frame feature count = {len(frame_features)}, expected 260"
        )

    return frame_features.astype(np.float32)


# ============================================================
# POSITIONAL ENCODING (SAME AS TRAINING)
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=30):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# TRANSFORMER (EXACT SAME ARCHITECTURE AS TRAINING)
# ============================================================

class ISLTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=SEQ_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD LABELS / MODEL / NORMALIZATION
# ============================================================

print("\nLoading labels...")
with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)
print("Labels:", labels)

print("\nLoading model...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "classes" in checkpoint:
    classes = checkpoint["classes"]
else:
    classes = labels

print("Classes:", classes)

if isinstance(checkpoint, dict) and "input_dim" in checkpoint:
    model_input_dim = checkpoint["input_dim"]
else:
    model_input_dim = INPUT_DIM

print("Model input dimension:", model_input_dim)

if model_input_dim != INPUT_DIM:
    raise ValueError(
        f"Model expects {model_input_dim} features, "
        f"but realtime uses {INPUT_DIM}"
    )

model = ISLTransformer(
    input_dim=INPUT_DIM,
    num_classes=len(classes),
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.3,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()
print("Transformer loaded successfully.")

mean = None
std = None

if isinstance(checkpoint, dict) and "mean" in checkpoint and "std" in checkpoint:
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    print("Normalization loaded.")
else:
    print("WARNING: Normalization not found.")

# ============================================================
# TEXT TO SPEECH
# Windows SAPI (pywin32): speaks asynchronously and a new letter
# CANCELS any older speech, so you never hear stale letters.
# Fallback: pyttsx3 worker that only keeps the LATEST letter.
# ============================================================

print("\nStarting TTS...")

speak = None   # function speak(letter)

try:
    import win32com.client

    _sapi = win32com.client.Dispatch("SAPI.SpVoice")
    _sapi.Rate = 0
    try:
        for v in _sapi.GetVoices():
            if "zira" in v.GetDescription().lower():
                _sapi.Voice = v
                break
    except Exception:
        pass

    def speak(letter):
        # 1 = async (don't block), 2 = purge any speech still playing
        _sapi.Speak(str(letter), 1 | 2)

    print("TTS engine ready (SAPI).")

except Exception as e:
    print("SAPI not available (", e, ") -> using pyttsx3 fallback.")
    import pyttsx3

    _latest = queue.Queue(maxsize=1)
    _tts_stop = threading.Event()

    def _tts_worker():
        while not _tts_stop.is_set():
            try:
                letter = _latest.get(timeout=0.1)
            except queue.Empty:
                continue
            if letter is None:
                break
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                try:
                    for voice in engine.getProperty("voices"):
                        name = (getattr(voice, "name", "") or "").lower()
                        if "zira" in name or "female" in name:
                            engine.setProperty("voice", voice.id)
                            break
                except Exception:
                    pass
                engine.say(str(letter))
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as err:
                print("TTS error:", err)

    _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    _tts_thread.start()

    def speak(letter):
        # drop any older waiting letter, keep only the newest
        try:
            _latest.get_nowait()
        except queue.Empty:
            pass
        try:
            _latest.put_nowait(letter)
        except queue.Full:
            pass

    print("TTS engine ready (pyttsx3).")

# ============================================================
# MEDIAPIPE HOLISTIC (SAME SETTINGS AS TRAINING)
# ============================================================

print("Starting MediaPipe...")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================
# WEBCAM
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    holistic.close()
    raise SystemExit

print("Webcam opened.")

# ============================================================
# STATE
# ============================================================

sequence = deque(maxlen=SEQ_LEN)
pred_history = deque(maxlen=STABLE_FRAMES)

last_spoken = None       # letter already spoken for the current gesture
no_hand_frames = 0
random_frames = 0        # consecutive frames: hand visible but model not confident
shown_label = ""         # last stable letter shown on screen

# ============================================================
# REALTIME LOOP
# ============================================================

print("\n" + "=" * 60)
print("REALTIME DETECTION STARTED")
print("=" * 60)
print("\nShow an ISL alphabet A-E. Each gesture is spoken ONCE.")
print("To repeat the same letter, lower your hands briefly and show it again.")
print("Press Q to quit.\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    left_ok = results.left_hand_landmarks is not None
    right_ok = results.right_hand_landmarks is not None

    cv2.putText(
        frame, "LEFT: YES" if left_ok else "LEFT: NO", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 0) if left_ok else (0, 0, 255), 2
    )
    cv2.putText(
        frame, "RIGHT: YES" if right_ok else "RIGHT: NO", (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 0) if right_ok else (0, 0, 255), 2
    )

    # ---- no-hand handling: lets the same letter be spoken again ----
    if not left_ok and not right_ok:
        no_hand_frames += 1
    else:
        no_hand_frames = 0

    if no_hand_frames >= NO_HAND_RESET_FRAMES:
        last_spoken = None
        pred_history.clear()
        random_frames = 0
        shown_label = ""

    # ---- features ----
    sequence.append(extract_frame_features(results))

    if len(sequence) == SEQ_LEN:

        seq_array = np.array(sequence, dtype=np.float32)

        if mean is not None:
            seq_array = (seq_array - mean) / std

        tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        confidence = confidence.item()
        predicted_label = str(classes[prediction.item()])

        # ---- live prediction display ----
        if confidence >= CONFIDENCE_THRESHOLD:
            live_text = f"{predicted_label} {confidence * 100:.1f}%"
        else:
            live_text = f"Uncertain {confidence * 100:.1f}%"

        cv2.putText(
            frame, live_text, (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )

        # ---- random gesture detection ----
        hand_visible = left_ok or right_ok
        known_letter = (
            confidence >= CONFIDENCE_THRESHOLD
            and predicted_label in SPEAK_LABELS
        )

        if hand_visible and not known_letter:
            random_frames += 1
        else:
            random_frames = 0

        if random_frames >= RANDOM_FRAMES:
            shown_label = RANDOM_TEXT
            if last_spoken != RANDOM_TEXT:
                print(f"Speaking: {RANDOM_TEXT}")
                speak(RANDOM_TEXT)
                last_spoken = RANDOM_TEXT

        # ---- stability check (only when a hand is actually visible) ----
        if hand_visible and known_letter:
            pred_history.append(predicted_label)
        else:
            pred_history.append(None)

        is_stable = (
            len(pred_history) == STABLE_FRAMES
            and pred_history[0] is not None
            and all(p == pred_history[0] for p in pred_history)
        )

        if is_stable:
            stable_label = pred_history[0]
            shown_label = stable_label

            # Speak ONCE when a new stable gesture appears
            if stable_label in SPEAK_LABELS and stable_label != last_spoken:
                print(f"Speaking: {stable_label} ({confidence * 100:.1f}%)")
                speak(stable_label)
                last_spoken = stable_label

    else:
        cv2.putText(
            frame, f"Collecting: {len(sequence)}/{SEQ_LEN}", (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    cv2.putText(
        frame, f"Stable: {shown_label}", (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2
    )
    cv2.putText(
        frame, f"Spoken: {last_spoken if last_spoken else '-'}", (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2
    )

    cv2.imshow("ISL Alphabet Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
holistic.close()

try:
    _latest.put_nowait(None)   # stops the pyttsx3 worker (only exists in fallback mode)
except Exception:
    pass

print("\nRealtime detection stopped.")