import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "speedtraficdetectionmodel_6classes.h5"
IMG_SIZE = 30
SEUIL_CONFIANCE = 0.90
CLASSES = ['20', '30', '50', '60', '70', 'STOP']

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé")
print("🔍 Input shape:", model.input_shape)

# ===============================
# PREPROCESS (STRICT TRAINING MATCH)
# ===============================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_roi(roi_bgr):
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = clahe.apply(roi_gray)

    img = Image.fromarray(roi_gray)
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
print("🎥 Caméra active — Q pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ===============================
    # RED COLOR MASK
    # ===============================
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
           cv2.inRange(hsv, lower_red2, upper_red2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ===============================
    # CONTOURS
    # ===============================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # ROI carré (important)
        size = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        x1 = max(cx - size // 2, 0)
        y1 = max(cy - size // 2, 0)
        x2 = min(x1 + size, frame.shape[1])
        y2 = min(y1 + size, frame.shape[0])

        roi = frame[y1:y2, x1:x2]

        try:
            img_input = preprocess_roi(roi)
            preds = model.predict(img_input, verbose=0)[0]
            cls_idx = np.argmax(preds)
            confidence = preds[cls_idx]

            label = CLASSES[cls_idx] if confidence >= SEUIL_CONFIANCE else "INCERTAIN"

            color = (0, 255, 0) if confidence >= SEUIL_CONFIANCE else (0, 0, 255)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output,
                        f"{label} ({confidence*100:.1f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        except Exception as e:
            pass

    cv2.imshow("Détection panneaux (Optimisée)", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Fin")
