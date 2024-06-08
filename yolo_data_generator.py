import cv2
import numpy as np
from utilities import calculate_iou, create_target_tensor
from tensorflow.keras.utils import Sequence


class YoloDataGenerator(Sequence):
    def __init__(self, image_shape, batch_size: int, grid_shape: tuple, num_anchors: int, num_samples=1000):
        self.image_shape = image_shape  # h x w x 3
        self.batch_size = batch_size
        self.grid_shape = grid_shape
        self.num_anchors = num_anchors
        self.classes = ['circle', 'square']
        self.num_classes = 2
        self.min_shape_size = min(self.image_shape[0], self.image_shape[1]) // 15
        self.max_shape_size = min(self.image_shape[0], self.image_shape[1]) // 5

        # split range of shape sizes evenly among anchors
        anchors = []
        anchor_diff = (self.max_shape_size - self.min_shape_size) / num_anchors
        grid_cell_size = [self.image_shape[0] / self.grid_shape[0], self.image_shape[1] / self.grid_shape[1]]
        for n in range(num_anchors):
            anchor_size = int(anchor_diff * (n + 0.5))
            anchors.append([anchor_size / grid_cell_size[1], anchor_size / grid_cell_size[0]])

        self.anchors = anchors
        self.num_samples = num_samples

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        batch_size_actual = min(self.batch_size, self.num_samples - index * self.batch_size)

        x = np.zeros((batch_size_actual, *self.image_shape), dtype=np.float32)
        y = np.zeros((batch_size_actual, *self.grid_shape, self.num_anchors, self.num_classes + 5), dtype=np.float32)

        for i in range(batch_size_actual):
            image, annotations = self.generate_image_and_annotations()
            x[i] = image
            # Assuming create_target_tensor is implemented elsewhere and correctly handles the conversion
            y[i] = create_target_tensor(self.image_shape[0], self.image_shape[1], self.grid_shape, self.num_anchors,
                                        self.classes, annotations, self.anchors)

        return x, y

    def generate_image_and_annotations(self):
        image = np.zeros(self.image_shape, dtype=np.float32)
        annotations = {label: [] for label in self.classes}
        num_shapes = np.random.randint(1, 5)  # Generate 1 to 4 shapes

        # while under the desired number of shapes, generate a new shape
        while sum([len(shapes) for shapes in annotations.values()]) < num_shapes:
            shape_type = np.random.choice(self.classes)
            size = np.random.randint(self.min_shape_size, self.max_shape_size)
            x = np.random.randint(size // 2, self.image_shape[1] - self.max_shape_size // 2)
            y = np.random.randint(size // 2, self.image_shape[0] - self.max_shape_size // 2)

            # don't let shapes overlap
            for (_x, _y, _size1, _size2) in annotations['circle'] + annotations['square']:
                if calculate_iou([x, y, size, size], [_x, _y, _size1, _size2]) > 0.01:
                    break
            else:

                if shape_type == 'circle':
                    self.generate_circle(image, size, x, y)
                elif shape_type == 'square':  # square
                    self.generate_square(image, size, x, y)
                else:
                    raise ValueError(f'Invalid shape type: {shape_type}')
                annotations[shape_type].append([x, y, size, size])

        return image / 255.0, annotations  # Normalize image to [0, 1]


def generate_square(image, size, x, y):
    cv2.rectangle(image,
                  (int(x - size / 2), int(y - size / 2)),  # (x1, y1)
                  (x + size // 2, y + size // 2),  # (x2, y2)
                  (255, 255, 255),  # color
                  -1)  # border thickness, -1 = filled

    cv2.rectangle(image,
                  (int(x - size // 2.3), int(y - size // 2.3)),
                  (int(x + size // 2.3), int(y + size // 2.3)),
                  (100, 100, 0),
                  -1)


def generate_circle(image, size, x, y):
    cv2.circle(image, (x, y), size // 2, (255, 255, 255), -1)
    cv2.circle(image, (x, y), int(size // 2.3), (100, 100, 0), -1)


def create_anchor_boxes(annotations, num_anchors):
    """
    Create the anchor boxes using KMeans clustering. Assumes bounding boxes are of the format (cx, cy, w, h)
    :param annotations:
    :param num_anchors:
    :return:
    """
    bboxes = []
    for image_annotations in annotations:
        for label, label_bboxes in image_annotations.items():
            for bbox in label_bboxes:
                bboxes.append([bbox[2], bbox[3]])

    # Convert the bounding box coordinates to a NumPy array
    bboxes = np.array(bboxes)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_anchors, random_state=0)
    kmeans.fit(bboxes)

    # Get the cluster centers as the anchor boxes
    anchor_boxes = kmeans.cluster_centers_

    return anchor_boxes
