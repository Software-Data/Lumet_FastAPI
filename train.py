from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import os

# Registra tus datasets en formato COCO
data_dir = "C:/Users/drife/Documents/Repositorios/Lumet_FastAPI/dataset"

register_coco_instances("lumet_train", {}, os.path.join(data_dir, "train/_annotations.coco.json"), os.path.join(data_dir, "train"))
register_coco_instances("lumet_valid", {}, os.path.join(data_dir, "valid/_annotations.coco.json"), os.path.join(data_dir, "valid"))

# Configuración del modelo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("lumet_train",)
cfg.DATASETS.TEST = ("lumet_valid",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # ajusta según GPU
cfg.SOLVER.MAX_ITER = 3000    # ajusta según tamaño dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # cambia si tienes más de una clase

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Entrenamiento
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
