import os
import zipfile
import shutil
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
BASE_DIR = "../data/"
NEW_DATA_DIR = os.path.join(BASE_DIR, "new_data/retrain/")
TRAIN_DIR = os.path.join(BASE_DIR, "train/")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation/")
MODEL_PATH = "../models/crop_disease_model.keras"
RETRAINED_MODEL_PATH = "../models/retrained_model.keras"

# Helper Functions
def extract_zip(zip_path, extract_to):
    """
    Extract the contents of a ZIP file.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            dest_path = os.path.join(extract_to, member)
            if os.path.exists(dest_path):
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                else:
                    os.remove(dest_path)
        zip_ref.extractall(extract_to)

def merge_datasets(new_data_dir, train_dir):
    """
    Merge new datasets into the training dataset directory.
    Handle existing files gracefully by overwriting them if necessary.
    """
    for class_name in os.listdir(new_data_dir):
        new_class_path = os.path.join(new_data_dir, class_name)
        train_class_path = os.path.join(train_dir, class_name)

        # Validate directory name
        if not os.path.isdir(new_class_path):
            print(f"Skipping invalid directory: {new_class_path}")
            continue

        # Ensure the training class directory exists
        os.makedirs(train_class_path, exist_ok=True)

        # Copy files from the new class directory to the training class directory
        for file_name in os.listdir(new_class_path):
            src_file_path = os.path.join(new_class_path, file_name)
            dest_file_path = os.path.join(train_class_path, file_name)

            try:
                if os.path.exists(dest_file_path):
                    print(f"File {file_name} already exists in {train_class_path}. Overwriting...")
                    os.remove(dest_file_path)

                shutil.copy2(src_file_path, dest_file_path)
            except Exception as e:
                print(f"Error copying file {file_name}: {e}")

def adjust_model_for_new_classes(model, num_classes):
    """
    Adjust the model's output layer to match the number of classes.
    """
    # Define input shape explicitly
    input_shape = model.input_shape[1:]
    inputs = Input(shape=input_shape)

    # Pass inputs through the existing model except the last layer
    x = model(inputs, training=False)
    x = Dense(num_classes, activation='softmax', name='new_output')(x)

    # Create a new model
    adjusted_model = Model(inputs=inputs, outputs=x)

    # Compile the adjusted model
    adjusted_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return adjusted_model

# Retraining Function
def retrain_model():
    """
    Load, adjust, and retrain the model on new data.
    """
    # Merge new data into the training dataset
    merge_datasets(NEW_DATA_DIR, TRAIN_DIR)

    # Load the existing model
    model = load_model(MODEL_PATH)

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Get the number of classes in the training dataset
    num_classes = train_generator.num_classes
    print(f"Number of classes in the dataset: {num_classes}")

    # Adjust the model for the new number of classes
    model = adjust_model_for_new_classes(model, num_classes)

    # Set a checkpoint to save the retrained model
    checkpoint = ModelCheckpoint(RETRAINED_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

    # Retrain the model
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[checkpoint]
    )
    print("Model retrained and saved.")

# Main Execution
if __name__ == "__main__":
    retrain_model()
