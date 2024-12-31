import cv2
import numpy as np

def preprocess_image(image, target_size=(48, 48)):
    """
    Preprocess image for emotion detection
    
    Args:
        image: Input image
        target_size: Size to resize the image to
        
    Returns:
        Preprocessed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, target_size)

    normalized = resized / 255.0

    preprocessed = normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return preprocessed

def detect_faces(image):
    """
    Detect faces in an image using OpenCV's Haar Cascade
    
    Args:
        image: Input image
        
    Returns:
        List of face coordinates (x, y, w, h)
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def get_emotion_label(prediction):
    """
    Convert model prediction to emotion label
    
    Args:
        prediction: Model output prediction
        
    Returns:
        String label of the emotion
    """
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[np.argmax(prediction)]
