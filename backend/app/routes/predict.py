# from fastapi import APIRouter, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from app.utils.image_utils import preprocess_image
# from app.model.model import DeepfakeModel
# from app.config import ALLOWED_EXTENSIONS
# import shutil
# import os
# import uuid
# import asyncio  # <-- Import asyncio

# router = APIRouter()
# model = DeepfakeModel()

# @router.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     filename = file.filename or ""
#     extension = filename.rsplit(".", 1)[-1].lower()

#     if extension not in ALLOWED_EXTENSIONS:
#         raise HTTPException(status_code=400, detail=f"File type '{extension}' not allowed. Please upload a .jpg, .jpeg or .png")

#     unique_filename = f"{uuid.uuid4()}.{extension}"
#     file_path = os.path.join("temp", unique_filename)
#     os.makedirs("temp", exist_ok=True)

#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         print(f"[INFO] Saved uploaded file to {file_path}")

#         image_array = preprocess_image(file_path)
#         result = model.predict(image_array)

#         # ⏳ Add delay here before returning response
#         await asyncio.sleep(3)  # Delay for 3 seconds

#         return JSONResponse(content={"prediction": result})

#     except Exception as e:
#         print(f"[ERROR] Prediction failed: {e}")
#         raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")

#     finally:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"[INFO] Temp file {file_path} deleted.")
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.utils.image_utils import preprocess_image
from app.model.model import DeepfakeModel
from app.config import ALLOWED_EXTENSIONS
import shutil
import os
import uuid
from PIL import Image
import io

router = APIRouter(prefix="/predict", tags=["predict"])
model = DeepfakeModel()

@router.post("/")
async def predict_image(file: UploadFile = File(...)):
    # Validate file extension
    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{extension}' not allowed. Please upload a .jpg, .jpeg, or .png")

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join("temp", unique_filename)
    os.makedirs("temp", exist_ok=True)

    try:
        # Validate image before saving
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image.verify()  # Check for valid image
        image.close()

        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        print(f"[INFO] Saved uploaded file to {file_path}")

        # Preprocess and predict
        image_array = preprocess_image(file_path)
        result = model.predict(image_array)
        return JSONResponse(content={"prediction": bool(result)})  # Convert int to bool

    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[INFO] Temp file {file_path} deleted.")
            except Exception as e:
                print(f"[ERROR] Failed to delete temp file {file_path}: {str(e)}")
