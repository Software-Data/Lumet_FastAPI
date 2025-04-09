import os
from io import BytesIO
from collections import Counter

import numpy as np
import cv2
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from ultralytics import YOLO

# --- Configuración de variables y conexión ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "test"
COLL_INPUT = "imagens"
COLL_OUTPUT = "imagenes_analizadas"

app = FastAPI(title="Inspección de Carrocerías con IA")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll_in = db[COLL_INPUT]
coll_out = db[COLL_OUTPUT]

# Cargar modelo YOLO (puedes usar "yolov8n.pt" para pruebas)
model = YOLO("./yolo/yolov8n.pt")

# --- Funciones Auxiliares ---

def calcular_color_principal(image: Image.Image) -> str:
    """Calcula el color dominante de la imagen reduciéndola para mayor eficiencia."""
    img_red = image.resize((50, 50))
    pixels = np.array(img_red).reshape(-1, 3)
    colores = Counter([tuple(p) for p in pixels])
    color_rgb = colores.most_common(1)[0][0]
    return f"RGB{color_rgb}"

def dividir_en_cuadricula(imagen, filas=15, columnas=15):
    """Divide la imagen en celdas de una cuadrícula definida por filas y columnas."""
    alto, ancho = imagen.shape[:2]
    tam_y = alto // filas
    tam_x = ancho // columnas
    secciones = []
    for y in range(filas):
        for x in range(columnas):
            x_ini = x * tam_x
            y_ini = y * tam_y
            x_fin = x_ini + tam_x
            y_fin = y_ini + tam_y
            secciones.append(((x, y), (x_ini, y_ini, x_fin, y_fin)))
    return secciones

def encontrar_cuadros_con_objetos(results, cuadricula):
    """Determina en qué celdas de la cuadrícula se encuentran las detecciones del modelo YOLO."""
    imperfecciones = set()
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.int().tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for (grid_x, grid_y), (xi, yi, xf, yf) in cuadricula:
                if xi <= cx <= xf and yi <= cy <= yf:
                    imperfecciones.add((grid_x, grid_y))
    return list(imperfecciones)

def marcar_cuadricula(imagen, imperfecciones, filas=15, columnas=15):
    """Dibuja la cuadrícula y resalta en rojo las celdas donde se detectaron imperfecciones."""
    draw = ImageDraw.Draw(imagen)
    ancho, alto = imagen.size
    paso_x = ancho // columnas
    paso_y = alto // filas
    # Dibujar líneas de la cuadrícula
    for x in range(columnas):
        x0 = x * paso_x
        draw.line([(x0, 0), (x0, alto)], fill="gray", width=1)
    for y in range(filas):
        y0 = y * paso_y
        draw.line([(0, y0), (ancho, y0)], fill="gray", width=1)
    # Dibujar rectángulos en celdas con imperfecciones
    for celda in imperfecciones:
        grid_x, grid_y = celda
        x0 = grid_x * paso_x
        y0 = grid_y * paso_y
        x1 = x0 + paso_x
        y1 = y0 + paso_y
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    return imagen

# --- Endpoint principal ---
@app.get("/analizar/{id}")
async def analizar_imagen(id: str):
    try:
        # Recuperar el documento de la imagen original
        imagen_doc = coll_in.find_one({"_id": ObjectId(id)})
        if not imagen_doc:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")
        
        # Leer la imagen (campo "imagen") desde el buffer
        buffer = imagen_doc["imagen"]
        imagen_pil = Image.open(BytesIO(buffer)).convert("RGB")
        imagen_np = np.array(imagen_pil)
        
        # Predicción con YOLO
        results = model.predict(imagen_np, verbose=False)
        
        # Dividir en cuadrícula y obtener las celdas con detecciones
        cuadricula = dividir_en_cuadricula(imagen_np)
        imperfecciones = encontrar_cuadros_con_objetos(results, cuadricula)
        
        # Marcar la imagen con la cuadrícula y las imperfecciones
        imagen_marcada = marcar_cuadricula(imagen_pil.copy(), imperfecciones)
        
        # Calcular el color dominante de la imagen original
        color_principal = calcular_color_principal(imagen_pil)
        
        # Convertir la imagen marcada a buffer (formato PNG)
        output_buffer = BytesIO()
        imagen_marcada.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        imagen_resultante = output_buffer.getvalue()
        
        # Guardar el resultado en MongoDB (almacenando el buffer resultante)
        result_doc = {
            "imagen_original_id": ObjectId(id),
            "color_dominante": color_principal,
            "imperfecciones": [{"x": x, "y": y} for x, y in imperfecciones],
            "imagen_resultado": imagen_resultante,
            "contentType": "image/png"
        }
        result_id = coll_out.insert_one(result_doc).inserted_id
        
        return JSONResponse(
            status_code=200,
            content={
                "mensaje": "Imagen analizada correctamente",
                "id_resultado": str(result_id),
                "color_dominante": color_principal,
                "imperfecciones_detectadas": len(imperfecciones),
                "coordenadas": [{"x": x, "y": y} for x, y in imperfecciones],
                "contentType": "image/png"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
