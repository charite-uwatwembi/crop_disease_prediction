import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Sequential

def build_model(num_classes):
    """
    Build the model using MobileNetV2 as the base.
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    return model

def load_trained_model(model_path):
    """
    Load the saved model.
    """
    return load_model(model_path)

# Entry point for testing
if __name__ == "__main__":
    print("Building the model...")
    num_classes = 15
    model = build_model(num_classes)
    model.summary() 

    model.save("models/test_model.h5")
    print("Model saved as test_model.h5")

    loaded_model = load_trained_model("../models/test_model.h5")
    print("Loaded model successfully")
    loaded_model.summary()
