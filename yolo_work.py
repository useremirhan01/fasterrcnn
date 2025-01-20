import cv2
from ultralytics import YOLO

# YOLO modelini yükle (kendi modelinizi kullanıyorsanız yolunu belirtin)
model = YOLO('must_best.pt')  # YOLOv8 nano modeli, dilerseniz custom model ekleyin

# Kamera başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı.")
        break

    # YOLO tahmini yap (conf = 0.3 eşiği ile)
    results = model.predict(frame, conf=0.6)  # Düşük eşik daha fazla algılama sağlar
    
    # Bounding box çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
