import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import create_data_generators

# Paths
data_dir = "../data/dataset"
model_path = "../models/crop_disease_model.keras"


# Add the parent directory of preprocess.py to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocess import create_data_generators


def evaluate_model():
    """
    Evaluate the trained model using the validation dataset.
    """
    try:
        # Validate paths
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create data generators (train, validation, test)
        print("Creating data generators...")
        _, val_data, _ = create_data_generators(data_dir)

        # Load the trained model
        print("Loading the model...")
        model = load_model(model_path)

        # Evaluate the model
        print("Evaluating the model...")
        loss, accuracy = model.evaluate(val_data)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Generate predictions
        print("Generating predictions...")
        val_data.reset()  # Ensure the generator is at the start
        predictions = model.predict(val_data)
        y_pred = predictions.argmax(axis=1)
        y_true = val_data.classes

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

# For direct execution
if __name__ == "__main__":
    evaluate_model()
