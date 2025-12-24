import cv2
import subprocess
import numpy as np

# Resolution
width, height = 640, 480

# Run libcamera-vid to output raw video to stdout
command = [
    'libcamera-vid',
    '-t', '0',           # run indefinitely
    '--inline',
    '--codec', 'yuv420',  # or mjpeg
    '-o', '-'            # output to stdout
]

pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

while True:
    # Read a frame from stdout (must calculate size depending on format)
    raw_frame = pipe.stdout.read(width*height*3)  # for RGB24
    if len(raw_frame) < width*height*3:
        break

    frame = np.frombuffer(raw_frame, dtype=np.uint8)
    frame = frame.reshape((height, width, 3))

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipe.terminate()
