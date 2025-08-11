import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO
import logging

# Detectron2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# --- Logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lumet")

# --- MongoDB ---
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

# --- Configurar Detectron2 ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cfg = get_cfg()
cfg.merge_from_file(os.path.join(BASE_DIR, "configs", "mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Cambia si tienes más clases
cfg.MODEL.WEIGHTS = os.path.join(BASE_DIR, "output", "model_final.pth")
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if cfg.DATASETS.TRAIN else MetadataCatalog.get("__unused")

# --- FastAPI ---
app = FastAPI(title="Inspección de Carrocerías con IA")

# --- Modelos Pydantic ---
class AnalisisRequest(BaseModel):
    id: str
    color_referencia: str  # Ejemplo: "#AABBCC"

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

# --- Endpoint para listar imágenes originales ---
@app.get("/imagenes")
def get_imagenes():
    imagenes = list(coll_in.find({}, {"imagen": 0}))
    for img in imagenes:
        img["_id"] = str(img["_id"])
    logger.info(f"{len(imagenes)} imágenes obtenidas desde la colección original")
    return imagenes

# --- Endpoint para listar imágenes analizadas ---
@app.get("/imagenes-analizadas")
def get_imagenes_analizadas():
    imagenes = list(coll_out.find({}, {"imagen_resultado": 0}))
    for img in imagenes:
        img["_id"] = str(img["_id"])
        img["imagen_original_id"] = str(img["imagen_original_id"])
    logger.info(f"{len(imagenes)} imágenes obtenidas desde la colección analizada")
    return imagenes

# --- Endpoint principal de análisis ---
from collections import Counter

@app.post("/analizar")
async def analizar(req: AnalisisRequest):
    try:
        doc = coll_in.find_one({"_id": ObjectId(req.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")

        # Abrir imagen (RGB)
        imagen_pil = Image.open(BytesIO(doc["imagen"])).convert("RGB")
        # Convertir a np.array BGR para Detectron2
        imagen_np = np.array(imagen_pil)[:, :, ::-1]

        # Calcular color dominante y contraste (opcional)
        color_dominante = calcular_color_principal_hex(imagen_pil)
        color_contraste = obtener_color_contraste(color_dominante)

        # Inferencia Detectron2
        outputs = predictor(imagen_np)
        instances = outputs["instances"].to("cpu")

        # Extraer info: máscaras, cajas, clases
        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy().astype(int)
        classes = instances.pred_classes.numpy()

        imperfecciones_detalles = []
        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            label = metadata.get("thing_classes", ["clase_desconocida"])[classes[i]]
            imperfecciones_detalles.append({
                "label": label,
                "bbox": bbox
            })

        # Visualizar resultados sobre la imagen
        v = Visualizer(imagen_np[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(instances)
        imagen_marcada = Image.fromarray(v.get_image()[:, :, ::-1])

        # Guardar imagen marcada en buffer
        buffer_out = BytesIO()
        imagen_marcada.save(buffer_out, format="PNG")
        buffer_out.seek(0)

        # Guardar resultado en MongoDB
        result_doc = {
            "imagen_original_id": ObjectId(req.id),
            "color_dominante": color_dominante,
            "color_referencia": req.color_referencia,
            "imperfecciones": imperfecciones_detalles,
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
                "imperfecciones_detectadas": len(imperfecciones_detalles),
                "detalles": imperfecciones_detalles,
                "contentType": "image/png"
            }
        )

    except Exception as e:
        logger.error("Error en el endpoint /analizar", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor")
