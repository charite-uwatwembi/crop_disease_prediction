import os
import shutil
import zipfile
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import load_trained_model
from retrain import extract_zip, merge_datasets, retrain_model
import numpy as np
from fastapi.responses import JSONResponse

# # Base directory (root directory of the project)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define paths relative to the root project directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
NEW_DATA_DIR = os.path.join(DATA_DIR, 'new_data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')


# Ensure required directories exist
os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)



# Initialize FastAPI
app = FastAPI()

# Load the pre-trained model
model_path = os.path.join(MODEL_DIR, "crop_disease_model.keras")
model = load_trained_model(model_path)





# Class labels mapping
CLASS_LABELS = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato__Target_Spot',
    12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy',
}
# Template rendering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of app.py
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route for the home page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for the retrain page
@app.get("/retrain.html", response_class=HTMLResponse)
async def read_retrain(request: Request):
    return templates.TemplateResponse("retrain.html", {"request": request})

# Route for the visualizations page
@app.get("/visualizations.html", response_class=HTMLResponse)
async def read_visualizations(request: Request):
    return templates.TemplateResponse("visualizations.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Display the home page with the upload form.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the class of the uploaded image and return class label, ID, and confidence.
    """
    try:
        # Read the uploaded file as bytes and wrap it in BytesIO
        contents = await file.read()
        image = BytesIO(contents)

        # Load and preprocess the image
        img = load_img(image, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class_id = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        class_label = CLASS_LABELS.get(predicted_class_id, "Unknown")

        # Return prediction as JSON
        return JSONResponse({
            "class_id": predicted_class_id,
            "class_label": class_label,
            "confidence": confidence,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/upload_retrain/")
async def upload_and_retrain(file: UploadFile = File(...)):
    """
    Upload a zip file and trigger retraining. If the dataset already exists, overwrite it.
    """
    try:
        file_name = file.filename.replace(" ", "_")
        zip_path = os.path.join(NEW_DATA_DIR, file_name)

        # Check if a file with the same name exists and remove it
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # Save the uploaded file
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract and merge data
        extract_zip(zip_path, NEW_DATA_DIR)
        merge_datasets(NEW_DATA_DIR, TRAIN_DIR)

        # Retrain the model
        retrain_model()
        return {"message": "Retraining completed successfully!"}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Utility function to extract zip files
def extract_zip(zip_path, extract_to):
    """
    Extract the contents of a zip file.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise

# Utility function to merge datasets
def merge_datasets(new_data_dir, train_dir):
    """
    Merge new datasets into the training dataset directory.
    """
    for class_name in os.listdir(new_data_dir):
        new_class_path = os.path.join(new_data_dir, class_name)
        train_class_path = os.path.join(train_dir, class_name)

        if not os.path.isdir(new_class_path):
            continue

        if os.path.exists(train_class_path):
            shutil.rmtree(train_class_path)

        shutil.copytree(new_class_path, train_class_path)
