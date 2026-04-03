from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import cv2

app = FastAPI(title="Gafas IA - Detección Objetos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo COCO-SSD
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

@app.get("/")
async def root():
    return {"message": "Gafas IA API lista", "endpoint": "/predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).resize((320, 320))
        img_array = np.array(image)[np.newaxis, ...].astype(np.uint8)
        
        results = model(img_array)
        result = {k: v.numpy() for k, v in results.items()}
        
        detections = []
        boxes = result['detection_boxes'][0]
        classes = result['detection_classes'][0]
        scores = result['detection_scores'][0]
        
        for i in range(len(scores)):
            if scores[i] > 0.5:
                detections.append({
                    "class": labels[int(classes[i])],
                    "confidence": float(scores[i]),
                    "bbox": boxes[i].tolist()
                })
        
        return {
            "status": "success",
            "detections": detections[:10]  # Máximo 10 detecciones
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)