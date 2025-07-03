from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from io import BytesIO
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from pymongo import MongoClient
from bson import ObjectId
from ultralytics import YOLO
import colorsys

# --- MongoDB ---
MONGO_URI = "mongodb://192.168.100.10:27017/"
client = MongoClient(MONGO_URI)
db = client["lumet"]
coll_in = db["imagens"]
coll_out = db["imagenes_analizadas"]

# --- App ---
app = FastAPI(title="Inspección de Carrocerías con IA")
model = YOLO("entrenamientos/imperfecciones_carroceria/weights/last.pt")

# --- Pydantic ---
class AnalisisRequest(BaseModel):
    id: str
    color_referencia: str  # formato: "#AABBCC"

# --- Utilidades ---

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb).upper()

def calcular_color_principal_hex(image: Image.Image) -> str:
    img_red = image.resize((50, 50))
    pixels = np.array(img_red).reshape(-1, 3)
    colores = Counter([tuple(p) for p in pixels])
    color_rgb = colores.most_common(1)[0][0]
    return rgb_to_hex(color_rgb)

def obtener_color_contraste(hex_color):
    r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    # Luminosidad según fórmula W3C
    luminancia = (0.299 * r + 0.587 * g + 0.114 * b)
    return "#000000" if luminancia > 186 else "#000000"

def dividir_en_cuadricula(imagen, filas=15, columnas=15):
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

def comparar_colores(c_referencia, c_dominante):
    def hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    r1, g1, b1 = hex_to_rgb(c_referencia)
    r2, g2, b2 = hex_to_rgb(c_dominante)
    distancia = np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
    max_dist = np.sqrt(255 ** 2 * 3)
    similitud = (1 - (distancia / max_dist)) * 100
    return max(0, round(similitud, 2))  # nunca menos de 0%

def marcar_cuadricula(imagen, imperfecciones, filas=15, columnas=15, color_cuadricula="#FF0000", color_celdas="#0000FF"):
    draw = ImageDraw.Draw(imagen)
    ancho, alto = imagen.size
    paso_x = ancho // columnas
    paso_y = alto // filas
    
    # Dibuja cuadrícula (líneas)
    for x in range(columnas + 1):
        x0 = x * paso_x
        draw.line([(x0, 0), (x0, alto)], fill=color_cuadricula, width=1)
    for y in range(filas + 1):
        y0 = y * paso_y
        draw.line([(0, y0), (ancho, y0)], fill=color_cuadricula, width=1)
    
    # Dibuja rectángulos en celdas con imperfecciones
    for grid_x, grid_y in imperfecciones:
        x0 = grid_x * paso_x
        y0 = grid_y * paso_y
        x1 = x0 + paso_x
        y1 = y0 + paso_y
        draw.rectangle([x0, y0, x1, y1], outline=color_celdas, width=3)
    return imagen

    draw = ImageDraw.Draw(imagen)
    ancho, alto = imagen.size
    paso_x = ancho // columnas
    paso_y = alto // filas
    for x in range(columnas):
        x0 = x * paso_x
        draw.line([(x0, 0), (x0, alto)], fill=color_cuadricula, width=1)
    for y in range(filas):
        y0 = y * paso_y
        draw.line([(0, y0), (ancho, y0)], fill=color_cuadricula, width=1)
    for grid_x, grid_y in imperfecciones:
        x0 = grid_x * paso_x
        y0 = grid_y * paso_y
        x1 = x0 + paso_x
        y1 = y0 + paso_y
        draw.rectangle([x0, y0, x1, y1], outline=color_cuadricula, width=2)
    return imagen

# --- Endpoint ---
@app.post("/analizar")
async def analizar(req: AnalisisRequest):
    try:
        doc = coll_in.find_one({"_id": ObjectId(req.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")

        buffer = doc["imagen"]
        imagen_pil = Image.open(BytesIO(buffer)).convert("RGB")
        imagen_np = np.array(imagen_pil)

        # Obtener color principal y contraste
        color_dominante = calcular_color_principal_hex(imagen_pil)
        color_contraste = obtener_color_contraste(color_dominante)
        similitud = comparar_colores(color_dominante, req.color_referencia)

        # Detección
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

        imagen_marcada = marcar_cuadricula(imagen_pil.copy(), imperfecciones_cuadricula, color_celdas=color_contraste)
        output_buffer = BytesIO()
        imagen_marcada.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        imagen_resultante = output_buffer.getvalue()

        result_doc = {
            "imagen_original_id": ObjectId(req.id),
            "color_dominante": color_dominante,
            "color_referencia": req.color_referencia,
            "similitud_color": similitud,
            "imperfecciones": imperfecciones_detalles,
            "cuadricula_afectada": [{"x": x, "y": y} for x, y in imperfecciones_cuadricula],
            "imagen_resultado": imagen_resultante,
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
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
