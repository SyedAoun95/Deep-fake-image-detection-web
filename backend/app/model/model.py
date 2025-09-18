import tensorflow as tf
from tensorflow.keras.models import load_model
from app.config import MODEL_PATH

class DeepfakeModel:
    def __init__(self):
        try:
            self.model = load_model(MODEL_PATH)
            print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def predict(self, image_array):
        try:
            prediction = self.model.predict(image_array, verbose=0)
            print(f"[INFO] Prediction raw output: {prediction}")
            threshold = 0.95  # ← Adjusted threshold
            result = int(prediction[0][0] > threshold)
            print(f"[INFO] Prediction result: {'Deepfake' if result else 'Authentic'}")
            return result
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
