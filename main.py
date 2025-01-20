import torch
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

def main():
    # Eğitim ve doğrulama veri setinin yolları
    train_img_dir = "datasets/train"
    train_ann = "datasets/annotations/instances_train.json"

    val_img_dir = "datasets/valid"
    val_ann = "datasets/annotations/instances_valid.json"

    # Veri setlerini COCO formatında kaydet
    register_coco_instances("my_dataset_train", {}, train_ann, train_img_dir)
    register_coco_instances("my_dataset_val", {}, val_ann, val_img_dir)

    # Config ayarları
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 11250
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # Veri kümenizdeki sınıf sayısı (arka plan hariç)

    # Model eğitimi
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluator ekleme
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    print("Evaluation results:")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    # Modeli kaydetme (pt formatında)
    torch.save(trainer.model.state_dict(), "faster_rcnn_model.pt")
    print("Model başarıyla kaydedildi: faster_rcnn_model.pt")

if __name__ == "__main__":
    main()
