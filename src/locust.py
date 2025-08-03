# locustfile.py

from locust import HttpUser, task, between
import random
import os

class PredictUser(HttpUser):
    wait_time = between(1, 2.5)  # Simulate user think time

    @task
    def predict_image(self):
        # Use a random image from a test sample folder
        test_folder = "sample_test_images"
        image_files = [f for f in os.listdir(test_folder) if f.endswith((".jpg", ".png"))]
        if not image_files:
            print("No test images found in sample_test_images/")
            return

        file_name = random.choice(image_files)
        with open(os.path.join(test_folder, file_name), "rb") as image_file:
            self.client.post(
                "/predict",
                files={"file": (file_name, image_file, "image/jpeg")}
            )
