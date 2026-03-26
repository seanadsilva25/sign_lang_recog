from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
import base64

app = Flask(__name__)

loaded = pickle.load(open("isl_model.pkl","rb"))
model = loaded["model"]
scaler = loaded["scaler"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   
    max_num_hands=2,
    min_detection_confidence=0.5
)

@app.route('/')
def home():
    return render_template('index4.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']

    img_bytes = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'gesture': 'Camera Error'})

    #frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        hand_vectors = []
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            single_hand = []
            
            for lm in hand_landmarks.landmark:
                single_hand.append(lm.x)
                single_hand.append(lm.y)
                single_hand.append(lm.z)

            hand_vectors.append(single_hand)

            # If only 1 hand → pad second hand with zeros
        while len(hand_vectors) < 2:
            hand_vectors.append([0.0] * 63)

    # Combine both hands → 126 features
        data = hand_vectors[0] + hand_vectors[1]

    # Convert + scale
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)

        prediction = model.predict(data)

        return jsonify({'gesture': prediction[0]})
   

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)