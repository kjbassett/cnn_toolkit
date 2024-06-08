import numpy as np
from utilities import create_target_tensor


class YoloDataConverter(Sequence):
    def __init__(self, images: list, annotations: list, batch_size: int, grid_shape: tuple, classes: list, anchors,
                 shuffle=True, augment=None):
        self.images = images
        self.image_shape = self.images[0].shape  # h x w x 3
        self.annotations = annotations  # But annotations are [x, y, w, h]
        self.batch_size = batch_size
        self.grid_shape = grid_shape  # h x w
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.classes = classes
        self.num_classes = len(classes)
        self.anchors = anchors
        self.shuffle = shuffle
        self.augment = augment

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size: (index + 1) * self.batch_size]
        batch_annotations = self.annotations[index * self.batch_size: (index + 1) * self.batch_size]

        x = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], 3))  # b x h x w x 3
        y = np.zeros((self.batch_size, self.grid_shape[0], self.grid_shape[1], self.num_anchors, self.num_classes + 5))

        for i, (image, annotations) in enumerate(zip(batch_images, batch_annotations)):
            if self.augment:
                image, annotations = self.augment(image, annotations)

            x[i] = image
            y[i] = create_target_tensor(self.image_shape[0], self.image_shape[1], self.grid_shape,
                                        self.classes, annotations, self.anchors)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.images))
            np.random.shuffle(indices)
            self.images = [self.images[i] for i in indices]
            self.annotations = [self.annotations[i] for i in indices]
