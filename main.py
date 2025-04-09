from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import cv2
import os

app = FastAPI()

# 🔐 Conexión a MongoDB Atlas
MONGO_URI = "TU_URI_DE_ATLAS"
client = MongoClient(MONGO_URI)
db = client["mi_base_de_datos"]
col_imagenes = db["imagenes"]
col_analisis = db["analisis"]

# 📦 Cargar modelo YOLOv8n
model = YOLO("yolov8n.pt")  # Asegúrate de tener este archivo en tu carpeta

@app.get("/analizar/{id_imagen}")
def analizar(id_imagen: str):
    try:
        oid = ObjectId(id_imagen)
    except:
        raise HTTPException(status_code=400, detail="ID inválido")

    doc = col_imagenes.find_one({"_id": oid})
    if not doc or "imagen" not in doc:
        raise HTTPException(status_code=404, detail="Imagen no encontrada o sin buffer")

    buffer = doc["imagen"]
    np_img = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=422, detail="Error al convertir la imagen")

    # 🧠 Análisis con YOLO
    results = model(img)

    imperfecciones = []
    for box in results[0].boxes:
        imperfecciones.append({
            "clase": model.names[int(box.cls[0])],
            "confianza": float(box.conf[0]),
            "coordenadas": [float(coord) for coord in box.xyxy[0].tolist()]
        })

    # 📝 Guardar resultado en otra colección
    col_analisis.insert_one({
        "id_imagen": oid,
        "resultado": imperfecciones,
        "fecha_analisis": datetime.now()
    })

    return {
        "mensaje": "Análisis completado",
        "id_imagen": str(oid),
        "resultado": imperfecciones
    }
