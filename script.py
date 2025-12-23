import cv2
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "speedtraficdetectionmodel_6classes.tflite"
IMG_SIZE = 30
SEUIL_CONFIANCE = 0.90
CLASSES = ['20', '30', '50', '60', '70', 'STOP']

# ===============================
# LOAD TFLITE MODEL
# ===============================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded")

# ===============================
# PREPROCESS
# ===============================
def preprocess_roi(roi_bgr):
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(roi_gray)
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_tflite(img_input):
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

# ===============================
# CAMERA LOOP (same logic)
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
print("🎥 Camera active — Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red mask
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
           cv2.inRange(hsv, lower_red2, upper_red2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]

        try:
            img_input = preprocess_roi(roi)
            preds = predict_tflite(img_input)

            cls_idx = np.argmax(preds)
            confidence = preds[cls_idx]

            label = CLASSES[cls_idx] if confidence >= SEUIL_CONFIANCE else "INCERTAIN"

            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output,
                        f"{label} ({confidence*100:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

        except:
            pass

    cv2.imshow("TFLite Traffic Sign Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
