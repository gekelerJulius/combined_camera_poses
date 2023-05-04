from typing import List
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import ndarray

from classes.bounding_box import BoundingBox

from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

matplotlib.use('TkAgg')  # or another backend


def visualize(image, bounding_boxes: List[BoundingBox], poses: ndarray):
    fig = plt.figure(figsize=(10, 5.2))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image)

    for box in bounding_boxes:
        x, y, w, h = box.min_x, box.min_y, box.get_width(), box.get_height()
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)

    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses[..., 1], poses[..., 2] = poses[..., 2], -poses[..., 1]
    for i, pose in enumerate(poses):
        pose_ax.scatter(*pose.T, s=2)

        for connection in POSE_CONNECTIONS:
            if i == connection[0]:
                other = poses[connection[1]]

                # Draw a line between the two points
                pose_ax.plot(
                    [pose[0], other[0]],
                    [pose[1], other[1]],
                    [pose[2], other[2]],
                    color='black'
                )

    fig.tight_layout()
    plt.savefig("test.png")
    plt.show()
