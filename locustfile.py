from locust import HttpUser, task, between
import os

class CropDiseasePredictionUser(HttpUser):
    wait_time = between(1, 5)  

    @task(1)
    def predict_endpoint(self):
        # Simulate a POST request to the predict endpoint
        with open(os.path.join("sample_images", "example_image.jpg"), "rb") as img:
            files = {"file": img}
            self.client.post("/predict", files=files)

    @task(1)
    def upload_and_retrain_endpoint(self):
        # Simulate a POST request to upload and retrain
        with open("new_data.zip", "rb") as data_zip:
            files = {"file": data_zip}
            self.client.post("/upload_retrain", files=files)
