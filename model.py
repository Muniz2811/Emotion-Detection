import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a very lightweight CNN model for emotion detection
    """
    K.clear_session()

    model = Sequential([
        Input(shape=input_shape),

        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    print("Model created successfully")
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model compiled successfully")
    print("\nModel Summary:")
    model.summary()
    
    return model
