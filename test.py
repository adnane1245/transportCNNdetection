import cv2

gst_pipeline = (
    "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 "
    "! videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
