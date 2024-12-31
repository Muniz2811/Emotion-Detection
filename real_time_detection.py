import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def detect_emotions():
    model = load_model('models/emotion_model_final.keras')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    cap = cv2.VideoCapture(0)
    
    print("Starting real-time emotion detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=[0, -1])

            prediction = model.predict(face_roi, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            emotion = emotions[emotion_idx]
            confidence = prediction[emotion_idx] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            text = f"{emotion}: {confidence:.1f}%"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detect_emotions()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
