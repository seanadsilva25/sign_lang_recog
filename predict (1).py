"""
╔══════════════════════════════════════════════════════════════╗
║     ISL Recognition — STAGE 3: Real-Time Prediction          ║
╚══════════════════════════════════════════════════════════════╝

Run this AFTER Stage 2 has produced isl_model.pkl.

What this script does:
  1. Loads the trained model (isl_model.pkl)
  2. Opens your webcam
  3. Detects hand landmarks in real time using MediaPipe
  4. Feeds landmarks into the model every frame
  5. Displays the predicted word on screen live

Press Q to quit.
"""

import cv2
import mediapipe as mp
import pickle
import numpy as np
from pathlib import Path
from collections import deque, Counter
from typing import Optional, List

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH        = "model/isl_model.pkl"   # trained model from Stage 2
NUM_LANDMARKS     = 21
COORDS            = 3
MAX_HANDS         = 2
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS          # 63
TOTAL_FEATURES    = MAX_HANDS * FEATURES_PER_HAND   # 126

# Smoothing: average predictions over this many frames to reduce flickering
SMOOTHING_WINDOW    = 20   # increased for more stable predictions
CONFIDENCE_THRESHOLD = 0.4  # show prediction if model is 40%+ sure


# ─────────────────────────────────────────────
#  STEP 1 — Load the trained model
# ─────────────────────────────────────────────
def load_model(model_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Run stage2_model_training.py first."
        )
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Support both old format (just model) and new format (dict with scaler)
    if isinstance(data, dict):
        model  = data["model"]
        scaler = data["scaler"]
    else:
        model  = data
        scaler = None

    print(f"[INFO] Model loaded from: {model_path}")
    return model, scaler


#  STEP 2 — Extract landmarks (same as Stage 1)
def extract_features(mediapipe_result) -> Optional[List[float]]:
    """
    Convert MediaPipe result into a flat 126-element feature vector.
    Returns None if no hand is detected.
    """
    if not mediapipe_result.multi_hand_landmarks:
        return None

    hand_vectors = []
    for hand_landmarks in mediapipe_result.multi_hand_landmarks[:MAX_HANDS]:
        flat = []
        for lm in hand_landmarks.landmark:
            flat.extend([lm.x, lm.y, lm.z])
        hand_vectors.append(flat)

    # Pad second hand slot with zeros if only one hand detected
    while len(hand_vectors) < MAX_HANDS:
        hand_vectors.append([0.0] * FEATURES_PER_HAND)

    return hand_vectors[0] + hand_vectors[1]


#  STEP 3 — Draw prediction UI on frame
def draw_ui(frame, predicted_word: str, confidence: float, hand_detected: bool):
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    if hand_detected and predicted_word:
        # Show predicted word
        cv2.putText(frame, predicted_word.upper(), (15, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 180), 2)
        # Show confidence
        conf_text = f"{confidence * 100:.1f}% confident"
        cv2.putText(frame, conf_text, (w - 210, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Show your hand...", (15, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 80, 255), 2)

    # Hand detection dot indicator
    dot_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.circle(frame, (w - 20, 20), 10, dot_color, -1)

    # Bottom hint
    cv2.putText(frame, "Press Q to quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


#  STEP 4 — Real-time prediction loop
def run_prediction(model, scaler):
    """
    Open webcam and predict ISL word in real time on every frame.
    """
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Webcam open. Show your hand to get a prediction.")
    print("[INFO] Press Q to quit.\n")

    # Smoothing buffer — stores recent predictions to reduce flickering
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    predicted_word    = ""
    confidence        = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = hands_model.process(rgb_frame)

        hand_detected = result.multi_hand_landmarks is not None

        # Draw hand skeleton
        if hand_detected:
            for hand_lms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS
                )

        # Extract features and predict
        features = extract_features(result)
        if features is not None:
            features_array = np.array(features).reshape(1, -1)   # shape: (1, 126)

            # Apply the same normalization used during training
            if scaler is not None:
                features_array = scaler.transform(features_array)

            # Get predicted class and its probability
            prediction  = model.predict(features_array)[0]
            proba       = model.predict_proba(features_array)[0]
            max_conf    = float(np.max(proba))

            # Only update prediction if model is confident enough
            if max_conf >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(prediction)

            # Use the most common prediction in the buffer (majority vote)
            if prediction_buffer:
                most_common    = Counter(prediction_buffer).most_common(1)[0][0]
                predicted_word = most_common
                confidence     = max_conf
            else:
                predicted_word = ""
                confidence     = 0.0
        else:
            # No hand — clear the buffer so stale predictions don't persist
            prediction_buffer.clear()
            predicted_word = ""
            confidence     = 0.0

        # Draw UI overlay
        frame = draw_ui(frame, predicted_word, confidence, hand_detected)
        cv2.imshow("ISL Prediction — Stage 3", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()


#  Entry point
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  ISL — STAGE 3: REAL-TIME PREDICTION")
    print("=" * 50 + "\n")

    model, scaler = load_model(MODEL_PATH)
    run_prediction(model, scaler)