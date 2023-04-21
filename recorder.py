# Record video from two cameras and save to file
import os

import cv2
import numpy as np

# Check if recordings folder exists
if not os.path.exists("recordings"):
    os.makedirs("recordings")

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# Define the mp4 codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out1 = cv2.VideoWriter("recordings/camera1.mp4", fourcc, 20.0, (640, 480))
out2 = cv2.VideoWriter("recordings/camera2.mp4", fourcc, 20.0, (640, 480))

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        # Write the frame into the file 'output.avi'
        out1.write(frame1)
        out2.write(frame2)

        cv2.imshow("frame1", frame1)
        cv2.imshow("frame2", frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release everything if job is finished
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
