import os
import shutil
from sklearn.model_selection import train_test_split

def split_for_retraining(train_dir, retrain_dir, retrain_split=0.2):
    """
    Reserve a portion of the training dataset for retraining.
    
    Parameters:
        train_dir (str): Path to the existing training data directory.
        retrain_dir (str): Path to the directory where retraining data will be saved.
        retrain_split (float): Fraction of training data to reserve for retraining.
    """
    # Get the class labels (subdirectory names)
    classes = os.listdir(train_dir)

    # Create the retrain directory if it doesn't exist
    os.makedirs(retrain_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(retrain_dir, cls), exist_ok=True)

    for cls in classes:
        class_dir = os.path.join(train_dir, cls)
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]

        # Split training data into retrain and remaining
        retrain_data, remaining_data = train_test_split(images, test_size=1-retrain_split, random_state=42)

        # Move files to respective directories
        def move_files(file_list, target_dir):
            for file in file_list:
                shutil.move(file, target_dir)

        # Move files for retraining
        move_files(retrain_data, os.path.join(retrain_dir, cls))

# Example usage
train_directory = 'data/dataset/train'
retrain_directory = 'data/dataset/retrain'
split_for_retraining(train_directory, retrain_directory)
