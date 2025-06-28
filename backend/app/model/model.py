# import tensorflow as tf
# # from keras.models import load_model
# from tensorflow.keras.models import load_model


# # Enable eager execution
# tf.compat.v1.enable_eager_execution()

# class DeepfakeModel:
#     def __init__(self):
#         self.model = load_model("app/model/efficientnet_deepfake.h5")


#     def predict(self, image_array):
#         prediction = self.model.predict(image_array)
#         return int(prediction[0][0] > 0.5)
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
            return int(prediction[0][0] > 0.5)
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")