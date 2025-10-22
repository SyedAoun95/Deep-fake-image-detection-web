# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from app.config import MODEL_PATH

# class DeepfakeModel:
#     def __init__(self):
#         try:
#             self.model = load_model(MODEL_PATH)
#             print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
#         except Exception as e:
#             raise Exception(f"Failed to load model: {str(e)}")

#     def predict(self, image_array):
#         try:
#             prediction = self.model.predict(image_array, verbose=0)
#             print(f"[INFO] Prediction raw output: {prediction}")
#             threshold = 0.95  # ← Adjusted threshold
#             result = int(prediction[0][0] > threshold)
#             print(f"[INFO] Prediction result: {'Deepfake' if result else 'Authentic'}")
#             return result
#         except Exception as e:
#             raise Exception(f"Prediction error: {str(e)}")
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score
import numpy as np
from app.config import MODEL_PATH

class DeepfakeModel:
    def __init__(self):
        try:
            self.model = load_model(MODEL_PATH)
            print(f"[INFO] Model loaded successfully from {MODEL_PATH}")

            # Optional: you can precompute precision & recall here if you have validation data
            self.precision = None
            self.recall = None
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def set_metrics(self, y_true, y_pred):
        """
        Optional method to compute dataset-level precision & recall once.
        Call this only during validation/testing, not per request.
        """
        try:
            self.precision = precision_score(y_true, y_pred)
            self.recall = recall_score(y_true, y_pred)
            print(f"[INFO] Precision: {self.precision:.4f}, Recall: {self.recall:.4f}")
        except Exception as e:
            print(f"[WARN] Could not compute metrics: {str(e)}")

    def predict(self, image_array):
        """
        Predict whether the image is deepfake or authentic.
        Returns label, confidence score, and optionally precision/recall.
        """
        try:
            # Get prediction probabilities
            prediction = self.model.predict(image_array, verbose=0)
            confidence = float(prediction[0][0])  # single value probability
            print(f"[INFO] Raw model output: {prediction}, confidence: {confidence:.4f}")

            # Apply threshold
            threshold = 0.5  # You can adjust based on calibration
            result = int(confidence > threshold)
            label = "Deepfake" if result == 1 else "Authentic"

            # Build response dictionary
            response = {
                "label": label,
                "confidence": round(confidence, 4)
            }

            # Optionally include precomputed metrics
            if self.precision is not None and self.recall is not None:
                response.update({
                    "precision": round(self.precision, 4),
                    "recall": round(self.recall, 4)
                })

            print(f"[INFO] Final prediction: {response}")
            return response

        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
