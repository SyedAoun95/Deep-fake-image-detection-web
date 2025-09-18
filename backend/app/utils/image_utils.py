# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.xception import preprocess_input

# def preprocess_image(image_file):
#     # ✅ Resize image to Xception input size
#     img = load_img(image_file, target_size=(299, 299))
    
#     # ✅ Convert to NumPy array
#     img = img_to_array(img)
    
#     # ✅ Add batch dimension
#     img = np.expand_dims(img, axis=0)
    
#     # ✅ Preprocess using Xception's method (scales input between -1 and 1)
#     img = preprocess_input(img)
    
#     return img

import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(image_path, target_size=(299, 299)):
    """
    Loads and preprocesses an image for deepfake detection.
    Resizes, converts to array, adds batch dimension, and normalizes.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (width, height).

    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values (0 to 1)
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")
