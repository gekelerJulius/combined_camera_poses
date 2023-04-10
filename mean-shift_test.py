import cv2
import time

# Load the video file
cap = cv2.VideoCapture("sim.mp4")

# Get the first frame
ret1, frame1 = cap.read()

if not ret1:
    raise Exception("Could not read video file")

last_frame = frame1

# Set the first frame's ROI as the object to track
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Remove pixels that are identical to the last frame
    calc_time = time.time()
    diff = cv2.absdiff(frame, last_frame)
    print(time.time() - calc_time)

    # Get YOLO bounding boxes and draw them on the current frame in green
    # bounding_boxes: List[BoundingBox] = getYoloBoundingBoxes(frame1)
    # for box in bounding_boxes:
    #     min_x, min_y, max_x, max_y = box.min_x, box.min_y, box.max_x, box.max_y
    #     cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Display the current frame with the tracking window
    cv2.imshow("frame", diff)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    last_frame = frame

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()
