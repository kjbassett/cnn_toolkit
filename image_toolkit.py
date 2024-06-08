import cv2
import numpy as np


def overlay_output(image, true_y, model_output, classes, objectness_threshold=0.1):
    grid_cell_h, grid_cell_w = (image.shape[0] / model_output.shape[0],
                                image.shape[1] / model_output.shape[1])
    # Draw grid lines
    for grid_x in range(model_output.shape[1]):
        cv2.line(image, (int(grid_x * grid_cell_w), 0), (int(grid_x * grid_cell_w), image.shape[0]), (20, 20, 20), 1)
    for grid_y in range(model_output.shape[0]):
        cv2.line(image, (0, int(grid_y * grid_cell_h)), (image.shape[1], int(grid_y * grid_cell_h)), (20, 20, 20), 1)

    while True:
        img_copy = image.copy()
        cv2.putText(img_copy,
                    f"Objectness Threshold: {objectness_threshold}",
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

        for color, data_source in [((0, 0, 255), true_y), ((0, 255, 0), model_output)]:
            for grid_y in range(model_output.shape[0]):
                for grid_x in range(model_output.shape[1]):
                    for anchor in range(model_output.shape[2]):
                        # Draw expected boxes
                        cx, cy, w, h, p, *class_probs = data_source[grid_y, grid_x, anchor]
                        if p < objectness_threshold:
                            continue
                        label_idx = np.argmax(class_probs)
                        label = classes[label_idx]
                        x1 = (grid_x + (cx - w/2)) * grid_cell_w
                        x2 = (grid_x + (cx + w/2)) * grid_cell_w
                        y1 = (grid_y + (cy - h/2)) * grid_cell_h
                        y2 = (grid_y + (cy + h/2)) * grid_cell_h

                        draw_box_and_label(img_copy, label, x1, x2, y1, y2, color=color, thickness=2)

        while True:
            cv2.imshow("Image", img_copy)
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                return
            elif key == ord("m"):
                objectness_threshold += 0.01
                break
            elif key == ord("n"):
                objectness_threshold -= 0.01
                break


def draw_box_and_label(image, label, x1, x2, y1, y2, color=(0, 255, 0), thickness=1):
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )
    if y1 < 10:
        label_y = y2 + 15
    else:
        label_y = y1 - 10
    cv2.putText(
        image,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        thickness,
    )
