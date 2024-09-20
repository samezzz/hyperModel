from fastapi import FastAPI
from app.routes import router
import tensorflow as tf

app = FastAPI()

# Register the routes from the routes file
app.include_router(router)

@app.get("/")
def status_gpu_check() -> dict[str, str]:
    gpu_msg = "Available" if tf.test.is_gpu_available() else "Unavailable"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }

