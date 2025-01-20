import cv2
import time
import torch
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def run_camera_inference(fps=30):
    # Detectron2 Config
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # Sınıf sayısı
    cfg.MODEL.WEIGHTS = "omar_best.pt"  # Model ağırlık dosyası
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Eşik değer 0.6 olarak güncellendi
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU varsa kullan

    # Predictor
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Kamera başlat
    cap = cv2.VideoCapture(0)  # Default kamera

    if not cap.isOpened():
        print("Kamera açılamadı.")
        return

    delay = 1 / fps  # Her kare arasındaki gecikme süresi
    print(f"{fps} FPS hızında çalışıyor. Çıkmak için 'q' tuşuna basın.")

    while True:
        start_time = time.time()

        # Kameradan görüntü al
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı, çıkılıyor...")
            break

        # Model ile çıkarım yap
        outputs = predictor(frame)

        # Bounding box çiz
        v = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Görüntüyü göster
        cv2.imshow("Real-time Faster R-CNN - 30 FPS", v.get_image()[:, :, ::-1])

        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # FPS'yi sınırlamak için uyku süresi ekle
        elapsed_time = time.time() - start_time
        sleep_time = max(delay - elapsed_time, 0)
        time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_inference(fps=30)
