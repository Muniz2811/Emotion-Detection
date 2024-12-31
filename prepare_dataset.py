import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def augment_image(image):
    """Apply random augmentation to an image"""
    angle = np.random.uniform(-15, 15)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    brightness = np.random.uniform(0.8, 1.2)
    image = np.clip(image * brightness, 0, 255)
    
    return image.astype(np.uint8)

def prepare_data(data_dir='data', max_samples_per_class=1000):
    """
    Prepare the dataset for training
    
    Args:
        data_dir: Directory containing the dataset
        max_samples_per_class: Maximum number of samples per emotion class
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    X = []
    y = []
    
    print("Loading and augmenting images...")

    class_counts = {}
    for i, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.exists(emotion_dir):
            files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg')]
            class_counts[emotion] = len(files)

    for i, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
        
        print(f"Processing {emotion} images...")
        files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg')]

        files = files[:max_samples_per_class]
        
        for file in tqdm(files):
            img_path = os.path.join(emotion_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img = cv2.resize(img, (48, 48))
                X.append(img)
                y.append(i)
        
        final_count = len([label for label in y if label == i])
        print(f"Final count for {emotion}: {final_count}")
    
    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], 48, 48, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nDataset prepared successfully:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    for i, emotion in enumerate(emotions):
        train_count = sum(y_train == i)
        test_count = sum(y_test == i)
        print(f"{emotion}: {train_count} training, {test_count} testing")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data()
