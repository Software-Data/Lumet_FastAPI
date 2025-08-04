from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from ultralytics import YOLO
from collections import Counter
from PIL import Image, ImageDraw
import numpy as np
import os
from io import BytesIO
import logging

# --- Configuración del logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lumet")

# --- Conexión a MongoDB ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
try:
    client.admin.command('ping')
    logger.info("Conexión a MongoDB realizada correctamente")
except Exception as e:
    logger.error("Error de conexión a MongoDB", exc_info=True)
    raise

db = client["lumet"]
coll_in = db["imagens"]
coll_out = db["imagenes_analizadas"]

# --- Modelo YOLO ---
model = YOLO("entrenamientos/imperfecciones_carroceria/weights/last.pt")

# --- App FastAPI ---
app = FastAPI(title="Inspección de Carrocerías con IA")

# --- Modelos ---
class AnalisisRequest(BaseModel):
    id: str
    color_referencia: str  # Ej. "#AABBCC"

# --- Utilidades ---
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb).upper()

def calcular_color_principal_hex(image: Image.Image) -> str:
    resized = image.resize((50, 50))
    pixels = np.array(resized).reshape(-1, 3)
    color_rgb = Counter(map(tuple, pixels)).most_common(1)[0][0]
    return rgb_to_hex(color_rgb)

def obtener_color_contraste(hex_color):
    r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    luminancia = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminancia > 186 else "#FFFFFF"

def dividir_en_cuadricula(img_np, filas=15, columnas=15):
    alto, ancho = img_np.shape[:2]
    return [((x, y), (x * ancho // columnas, y * alto // filas,
                     (x+1) * ancho // columnas, (y+1) * alto // filas))
            for y in range(filas) for x in range(columnas)]

def encontrar_cuadros_con_objetos(results, cuadricula):
    imperfecciones = set()
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.int().tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for (gx, gy), (xi, yi, xf, yf) in cuadricula:
                if xi <= cx <= xf and yi <= cy <= yf:
                    imperfecciones.add((gx, gy))
    return list(imperfecciones)

def comparar_colores(hex1, hex2):
    def hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    distancia = np.linalg.norm([r1 - r2, g1 - g2, b1 - b2])
    return max(0, round((1 - distancia / (255 * (3**0.5))) * 100, 2))

def marcar_cuadricula(imagen, imperfecciones, filas=15, columnas=15,
                      color_cuadricula="#FF0000", color_celdas="#0000FF"):
    draw = ImageDraw.Draw(imagen)
    ancho, alto = imagen.size
    paso_x = ancho // columnas
    paso_y = alto // filas

    for x in range(columnas + 1):
        draw.line([(x * paso_x, 0), (x * paso_x, alto)], fill=color_cuadricula, width=1)
    for y in range(filas + 1):
        draw.line([(0, y * paso_y), (ancho, y * paso_y)], fill=color_cuadricula, width=1)

    for gx, gy in imperfecciones:
        x0, y0 = gx * paso_x, gy * paso_y
        x1, y1 = x0 + paso_x, y0 + paso_y
        draw.rectangle([x0, y0, x1, y1], outline=color_celdas, width=3)

    return imagen

# --- Endpoints ---
@app.get("/imagenes")
def get_imagenes():
    imagenes = list(coll_in.find({}, {"imagen": 0}))
    for img in imagenes:
        img["_id"] = str(img["_id"])
    logger.info(f"{len(imagenes)} imágenes obtenidas desde la colección original")
    return imagenes

@app.get("/imagenes-analizadas")
def get_imagenes_analizadas():
    imagenes = list(coll_out.find({}, {"imagen_resultado": 0}))
    for img in imagenes:
        img["_id"] = str(img["_id"])
        img["imagen_original_id"] = str(img["imagen_original_id"])
    logger.info(f"{len(imagenes)} imágenes obtenidas desde la colección analizada")
    return imagenes

@app.post("/analizar")
async def analizar(req: AnalisisRequest):
    try:
        doc = coll_in.find_one({"_id": ObjectId(req.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")

        imagen_pil = Image.open(BytesIO(doc["imagen"])).convert("RGB")
        imagen_np = np.array(imagen_pil)

        color_dominante = calcular_color_principal_hex(imagen_pil)
        color_contraste = obtener_color_contraste(color_dominante)
        similitud = comparar_colores(color_dominante, req.color_referencia)

        results = model.predict(imagen_np, verbose=False)
        cuadricula = dividir_en_cuadricula(imagen_np)
        imperfecciones_cuadricula = encontrar_cuadros_con_objetos(results, cuadricula)

        imperfecciones_detalles = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                imperfecciones_detalles.append({
                    "label": label,
                    "bbox": [x1, y1, x2, y2]
                })

        imagen_marcada = marcar_cuadricula(imagen_pil.copy(), imperfecciones_cuadricula,
                                           color_celdas=color_contraste)
        buffer_out = BytesIO()
        imagen_marcada.save(buffer_out, format="PNG")
        buffer_out.seek(0)

        result_doc = {
            "imagen_original_id": ObjectId(req.id),
            "color_dominante": color_dominante,
            "color_referencia": req.color_referencia,
            "similitud_color": similitud,
            "imperfecciones": imperfecciones_detalles,
            "cuadricula_afectada": [{"x": x, "y": y} for x, y in imperfecciones_cuadricula],
            "imagen_resultado": buffer_out.getvalue(),
            "contentType": "image/png"
        }

        result_id = coll_out.insert_one(result_doc).inserted_id

        return JSONResponse(
            status_code=200,
            content={
                "mensaje": "Imagen analizada correctamente",
                "id_resultado": str(result_id),
                "color_dominante": color_dominante,
                "color_referencia": req.color_referencia,
                "similitud_color_%": similitud,
                "imperfecciones_detectadas": len(imperfecciones_detalles),
                "detalles": imperfecciones_detalles,
                "cuadricula_afectada": result_doc["cuadricula_afectada"],
                "contentType": "image/png"
            }
        )

    except Exception as e:
        logger.error("Error en el endpoint /analizar", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor")
