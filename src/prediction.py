from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import load_trained_model
import numpy as np

# Load the trained model
model_path = "../models/crop_disease_model.keras"  # Ensure this path is correct
model = load_trained_model(model_path)

def predict(image_file):
    """
    Predict the class of the uploaded image.
    """
    try:
        # Read and preprocess the uploaded image
        img = load_img(image_file, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {"class_id": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
