import cv2
import os
import numpy as np
import json
from utilities import load_annotations


class ImageAnnotator:

    def __init__(self, data_dir, labels, output_file):
        self.img = None
        self.img_copy = None
        self.data_dir = data_dir
        self.label = labels[0]
        self.labels = labels
        self.output_file = output_file
        self.annotations = {}
        self.current_image = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.delete_me = 0
        self.last_mouse_x, self.last_mouse_y = 0, 0

    def draw_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.rectangle(self.img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            else:
                self.last_mouse_x, self.last_mouse_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

            if self.ix > x:
                self.ix, x = x, self.ix
            if self.iy > y:
                self.iy, y = y, self.iy
            x = min(self.img.shape[1], max(0, x))
            y = min(self.img.shape[0], max(0, y))
            self.ix = min(self.img.shape[1], max(0, self.ix))
            self.iy = min(self.img.shape[0], max(0, self.iy))

            cv2.rectangle(self.img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            bbox = [self.ix, self.iy, x, y]
            self.annotations.setdefault(self.current_image, {})
            self.annotations[self.current_image].setdefault(self.label, [])
            self.annotations[self.current_image][self.label].append(bbox)

    def draw_overlay(self):
        self.draw_instructions()
        self.draw_crosshair()
        self.draw_bboxes()
        self.draw_current_label()

    def draw_bboxes(self):
        for label, bboxes in self.annotations.get(self.current_image, {}).items():
            for bbox in bboxes:
                cv2.rectangle(
                    self.img_copy,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2,
                )
                if bbox[1] < 10:
                    label_y = bbox[3] + 15
                else:
                    label_y = bbox[1] - 10
                cv2.putText(
                    self.img_copy,
                    label,
                    (bbox[0], label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    def annotate_images(self, skip_completed=False):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.draw_bbox)
        self.annotations = load_annotations(self.data_dir)
        img_index = 0
        files = [img for img in os.listdir(self.data_dir) if not img.endswith(".json")]
        while True:
            img_index = img_index % len(files)
            img_name = files[img_index]
            if img_name in self.annotations:
                if all([label in self.annotations[img_name] for label in self.labels]):
                    if skip_completed:
                        print(f"Skipping {img_name} (already annotated)")
                        continue

            img_path = os.path.join(self.data_dir, img_name)
            self.img = cv2.imread(img_path)
            self.current_image = img_name

            while True:
                self.img_copy = np.zeros((self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8)
                self.img_copy[:, :, 0:3] = self.img
                self.img_copy[:, :, 3] = 255
                self.draw_overlay()
                key = cv2.waitKey(25) & 0xFF
                if key == ord("z"):
                    self.annotations[self.current_image][self.label] = self.annotations[
                        self.current_image
                    ][self.label][:-1]
                if key == ord("n"):
                    img_index += 1
                    break
                if key == ord("b"):
                    img_index -= 1
                    break
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                if key == ord("t"):
                    self.label = self.labels[(self.labels.index(self.label) - 1) % len(self.labels)]
                if key == ord("y"):
                    self.label = self.labels[(self.labels.index(self.label) + 1) % len(self.labels)]
                cv2.imshow("Image", self.img_copy)


        cv2.destroyAllWindows()

    def save_annotations(self):
        with open(os.path.join(self.data_dir, self.output_file), "w") as f:
            json.dump(self.annotations, f, indent=4)

    def run(self, skip_completed=True):
        self.annotate_images(skip_completed=skip_completed)
        self.save_annotations()
        print(f"Annotations saved to {self.output_file}")

    def draw_current_label(self):
        # print label on screen on the corner furthest from the mouse
        # get current mouse position
        x, y = 300, 50

        cv2.putText(
            self.img_copy,
            self.label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 50, 50),
            3,
        )

    def draw_instructions(self):
        keys = ["q", "z", "n", "b", "t", "y"]
        descs = ["Quit", "Undo", "Next Image", "Previous Image", "Next Label", "Previous Label"]
        x1 = 300
        y1 = 0
        x2 = x1 + 400
        y2 = y1 + 100 + 40 * len(keys)

        alpha = 0.3
        overlay = np.zeros(self.img_copy.shape, dtype=np.uint8)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)

        self.img_copy = cv2.addWeighted(overlay, alpha, self.img_copy, 1, 0)

        y = 100
        for letter, instruction in zip(keys, descs):
            cv2.putText(
                self.img_copy,
                f"{letter}: {instruction}",
                (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                (0, 0, 0),
                3,
            )
            y += 40

    def draw_crosshair(self):
        cv2.line(self.img_copy, (0, self.last_mouse_y), (self.img.shape[1], self.last_mouse_y), (0, 255, 0), 1)
        cv2.line(self.img_copy, (self.last_mouse_x, 0), (self.last_mouse_x, self.img.shape[0]), (0, 255, 0), 1)