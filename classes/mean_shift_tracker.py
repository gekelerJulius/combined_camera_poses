from typing import Tuple

import cv2
from numpy import ndarray

from classes.bounding_box import BoundingBox


class MeanShiftTracker:
    img: ndarray
    box: BoundingBox
    track_window: Tuple[int, int, int, int]
    roi_hist: ndarray
    term_crit: Tuple[int, float, float]

    def __init__(self, img: ndarray, box: BoundingBox):
        self.img = img
        self.box = box
        box.increase_size(0.2)
        self.track_window = box.as_tracking_window()
        roi = box.crop_image(img)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, (0., 60., 32.), (180., 255., 255.))
        hist_roi = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)
        self.roi_hist = hist_roi.reshape(-1)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def track(self, img: ndarray) -> None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
