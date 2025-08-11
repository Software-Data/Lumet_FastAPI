import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

# Ruta a tu dataset (ajusta esta ruta a tu proyecto)
data_dir = "dataset"

# Registra los datasets en formato COCO
register_coco_instances("lumet_train", {}, os.path.join(data_dir, "train/_annotations.coco.json"), os.path.join(data_dir, "train"))
register_coco_instances("lumet_valid", {}, os.path.join(data_dir, "valid/_annotations.coco.json"), os.path.join(data_dir, "valid"))

# Configuración del modelo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("lumet_train",)
cfg.DATASETS.TEST = ("lumet_valid",)
cfg.DATALOADER.NUM_WORKERS = 2

# Usar pesos preentrenados del model zoo como punto de partida
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000  # Ajusta según tu dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Cambia si tienes más clases

# Carpeta donde se guardarán checkpoints y resultados
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Entrenador y entrenamiento
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Guardar modelo final (opcional, suele guardarse automáticamente)
trainer.checkpointer.save("model_final")
print(f"Modelo guardado en: {os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')}")
