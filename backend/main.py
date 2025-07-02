# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app.routes import predict
# import os

# app = FastAPI(title="Image Deepfake Detection API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or specify your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(predict.router)

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))  # Use Render's port or 8000 locally
#     uvicorn.run("main:app", host="0.0.0.0", port=port)
# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict
import os

app = FastAPI(title="Image Deepfake Detection API")

# ✅ Allow local frontend access (Vite typically runs on 5173, not 8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or "*" if debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register your prediction route
app.include_router(predict.router)

# ✅ Optional if you want to run directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
