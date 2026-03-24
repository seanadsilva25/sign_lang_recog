import cv2
import mediapipe as mp
import csv

label = input("Enter gesture label (hello/yes/no): ")#just for training phase

#loads hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

#used to open cam
cap = cv2.VideoCapture(0)

with open('dataset.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()#it reads live video framw

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                data = []
                
                for lm in hand_landmarks.landmark:
                    data.append(lm.x)
                    data.append(lm.y)
                    data.append(lm.z)

                data.append(label)
                writer.writerow(data)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
