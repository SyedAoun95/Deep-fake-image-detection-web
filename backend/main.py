from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict
import os

app = FastAPI(title="Image Deepfake Detection API")

# ✅ Allow frontend running on http://localhost:8080 and http://127.0.0.1:8080
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://192.168.56.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register your prediction router
app.include_router(predict.router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
