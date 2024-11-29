import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def preprocess_image(image_path, img_size=(128, 128)):
    """
    Resize and normalize a single image.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return tf.expand_dims(img_array, 0)  # Add batch dimension

def create_data_generators(data_dir, img_size=(128, 128), batch_size=32):
    """
    Create ImageDataGenerators for training, validation, and test data.
    """
    datagen_args = dict(
        rescale=1.0 / 255.0, 
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_datagen = ImageDataGenerator(**datagen_args, validation_split=0.1)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )

    validation_data = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    test_data = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    return train_data, validation_data, test_data

# Testing block
if __name__ == "__main__":
    # Replace this with the full path to your dataset
    data_directory = "D:/Study/BSE/MachineLearning/crop_disease_prediction/data/dataset/"
    
    # Verify that the directory exists
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"The directory {data_directory} does not exist.")
    
    img_size = (128, 128)
    batch_size = 32

    print("Creating data generators...")
    train_data, validation_data, test_data = create_data_generators(data_directory, img_size, batch_size)

    print("Sample from training data:")
    for images, labels in train_data:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
