

# Crop Disease Prediction

This project aims to help farmers and agricultural experts efficiently detect crop diseases using machine learning models trained on image datasets. The system allows users to upload images of crops to predict potential diseases and includes functionalities for retraining the model with new data.

---

## 📹 Demo Video
Watch the video demo here: [https://www.loom.com/share/53c5f26eade0436aa6c21eb022425748?sid=5139eea6-98fb-4a58-9c7c-fab2457117c6]()


---

## 📄 Project Description

The **Crop Disease Prediction** project leverages machine learning and image processing to classify crop diseases.  
Features include:
1. **Prediction**: Upload crop images to detect diseases.
2. **Retraining**: Add new images via a zip file to improve the model.
3. **Visualization**: View statistical insights and predictions.
4. **Scalability**: Simulate high traffic using Locust and evaluate performance.

This project was developed using **Python**, **FastAPI**, **TensorFlow/Keras**, and **Docker** for deployment.

---

## 🖼️ Screenshots

### **Home Page**

![Home Page Screenshot](/screenshots/home-phone.png)

### **Prediction Page**
![Home Page Screenshot](/screenshots/home-dec.png)

### **Retraining Page**
![Retraining Page Screenshot](/screenshots/retrain-dec.png)

### **Visualization Page**
![Visualization Page Screenshot](/screenshots/visualization.png)
![Retrain successful Screenshot](/screenshots/retrain.png)

---

## 🚀 How to Set Up the Project

### **1. Clone the Repository**
```bash
git clone https://github.com/charite-uwatwembi/crop_disease_prediction.git
cd crop_disease_prediction
```

### **2. Install Prerequisites**
Make sure you have the following installed:
- Python 3.8+
- pip
- Git
- Docker (if deploying using Docker)

### **3. Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### **4. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **5. Run the Application**
Start the FastAPI server:
```bash
python src/app.py
```
Access the app locally at `http://127.0.0.1:8000`.

### **6. Using Docker**
Build and run the Docker container:
```bash
docker build -t crop-disease-app .
docker run -p 8000:8000 -v D:/path/to/data:/app/data crop-disease-app
```

---

## 🧪 Locust Flood Test Simulation

To simulate high traffic and test the app's scalability:

### **1. Install Locust**
```bash
pip install locust
```

### **2. Run Locust**
Run the Locust file to start a load test:
```bash
locust -f locustfile.py
```
Access Locust at `http://127.0.0.1:8089` and configure the number of users and spawn rate.

### **3. Results**
Results of the flood request simulation will appear here after testing:
- Average Latency: _to be added_
- Success Rate: _to be added_
- Maximum Requests Handled: _to be added_

---

## 📊 Results and Insights

### **Prediction Results**
- Model accuracy: _to be added_
- Example predictions:
  - Image: Healthy Crop → Predicted as "Healthy"
  - Image: Infected Crop → Predicted as "Disease X"

### **Flood Test Results**
- Average response time under X users: _to be added_
- Maximum requests handled without failure: _to be added_

---

Feel free to contribute or raise issues in the repository! 
