import json
import os
import numpy as np
from sklearn.cluster import KMeans


def load_dataset(data_dir, resize=None, only_annotated=False, normalize=True, bbox_format="yolo"):
    """
    loads a list of images, a list of dictionaries of annotations, and a list of unique labels
    :param data_dir: folder containing images and annotations.json
    :param resize: (new_width, new_height) of image. None for no resizing. (width, height) because PIL likes it that way
    :param only_annotated: If True, only include images that have annotations
    :param normalize: If normalize == True, images will be in divided by 255
    :param bbox_format: corners=(x1, y1, x2, y2), yolo = (center_x, center_y, w, h)
    :return: images, annotations, labels
    """
    from PIL import Image
    # TODO what if > # of anchor boxes?
    annotations_dict = load_annotations(data_dir, bbox_format=bbox_format)
    images = []
    annotations = []
    labels = []  # list of unique labels

    for f in os.listdir(data_dir):
        if not f.endswith((".jpg", ".png")):
            continue
        if only_annotated and f not in annotations_dict:
            continue

        # image
        img = Image.open(os.path.join(data_dir, f))
        image_annotations = annotations_dict.get(f, {})
        if resize:
            ratio_x = resize[0] / img.size[0]  # img isn't a numpy array yet so 0 index holds the width
            ratio_y = resize[1] / img.size[1]  # img isn't a numpy array yet so 1 index holds the height

            img = img.resize(resize)

            resized_annotations = {}
            for label, bboxes in image_annotations.items():
                resized_annotations[label] = []
                for bbox in bboxes:
                    resized_annotations[label].append([
                        bbox[0] * ratio_x,
                        bbox[1] * ratio_y,
                        bbox[2] * ratio_x,
                        bbox[3] * ratio_y
                    ])
                    # print('Resized bbox: ', resized_annotations[label][-1])
            image_annotations = resized_annotations
        annotations.append(image_annotations)

        img = np.array(img)

        if normalize:
            img = img / 255
        images.append(img)

        # labels
        for label in image_annotations:
            if label not in labels:
                labels.append(label)

    return images, annotations, labels


def load_annotations(data_dir, bbox_format="corners"):
    """
    Converts from a
    :param data_dir:
    :param bbox_format: "corners" or "yolo". Data is stored as corners, yolo is center_x, center_y, width, height
    :return:
    """
    path = os.path.join(data_dir, 'annotations.json')
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        annotations = json.load(f)

    # if format == corners, already in correct format
    if bbox_format == "yolo":
        for img in annotations:
            for label in annotations[img]:
                for i, bbox in enumerate(annotations[img][label]):
                    annotations[img][label][i] = [
                        bbox[0] + (bbox[2] - bbox[0]) / 2,
                        bbox[1] + (bbox[3] - bbox[1]) / 2,
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ]
    return annotations


def calculate_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) score between two bounding boxes.
    Bounding boxes are expected to be in the format (x, y, w, h), where (x, y) is the
    center of the box, and w and h are the width and height of the box.
    :param bbox1: Numpy array of shape (4,), representing (x, y, w, h) for the first box
    :param bbox2: Numpy array of shape (4,), representing (x, y, w, h) for the second box
    :return: IoU score
    """
    # if a bbox is passed in with length 2, assume it's [w, h] and set x, y equal to the other box
    if len(bbox1) == 2 and len(bbox2) == 4:
        bbox1 = np.array([bbox2[0], bbox2[1], bbox1[0], bbox1[1]])
    elif len(bbox1) == 4 and len(bbox2) == 2:
        bbox2 = np.array([bbox1[0], bbox1[1], bbox2[0], bbox2[1]])
    elif len(bbox1) == 2 and len(bbox2) == 2:
        bbox1 = np.array([0, 0, bbox1[0], bbox1[1]])
        bbox2 = np.array([0, 0, bbox2[0], bbox2[1]])

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    bbox1_x1 = bbox1[0] - bbox1[2] / 2.0
    bbox1_y1 = bbox1[1] - bbox1[3] / 2.0
    bbox1_x2 = bbox1[0] + bbox1[2] / 2.0
    bbox1_y2 = bbox1[1] + bbox1[3] / 2.0

    bbox2_x1 = bbox2[0] - bbox2[2] / 2.0
    bbox2_y1 = bbox2[1] - bbox2[3] / 2.0
    bbox2_x2 = bbox2[0] + bbox2[2] / 2.0
    bbox2_y2 = bbox2[1] + bbox2[3] / 2.0

    # Calculate intersection coordinates
    inter_x1 = np.maximum(bbox1_x1, bbox2_x1)
    inter_y1 = np.maximum(bbox1_y1, bbox2_y1)
    inter_x2 = np.minimum(bbox1_x2, bbox2_x2)
    inter_y2 = np.minimum(bbox1_y2, bbox2_y2)

    # Calculate intersection area
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Calculate areas of each bounding box
    bbox1_area = (bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1)
    bbox2_area = (bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1)

    # Calculate union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou


def create_target_tensor(image_height: int, image_width: int, grid_shape: tuple, classes: list,
                         annotations: dict, anchors: np.ndarray):
    """
    Create a target tensor for YOLO loss calculation.

    Args:
        image_height (int): Height of the image.
        image_width (int): Width of the image.
        grid_shape (tuple): Shape of the grid (height, width).
        classes (int): All objects that the detector will train on.
        annotations (dict): dictionary {class: [bboxes]}. bboxes are of the format [x, y, width, height]
        anchors (numpy.ndarray): Array of anchor boxes.

    Returns:
        numpy.ndarray: Target tensor of shape (height, width, num_anchors, num_classes + 5).
    """
    height, width = grid_shape
    target_tensor = np.zeros((height, width, len(anchors), len(classes) + 5))

    for label, bboxes in annotations.items():
        class_idx = classes.index(label)  # Convert the label to an integer class index

        for bbox in bboxes:
            assign_bbox(anchors, bbox, class_idx, grid_shape, image_height, image_width, target_tensor)

    return target_tensor


def assign_bbox(anchors, bbox, class_idx, grid_shape, image_height, image_width, target_tensor):
    # Get the grid cells that the bounding box overlaps with
    grid_cell, relative_bbox = get_grid_cell_and_relative_bbox(bbox, grid_shape, image_height, image_width)
    grid_y, grid_x = grid_cell
    # Choose the best anchor box for the cropped bounding box
    anchor_idx = get_best_anchor(relative_bbox, anchors)
    # Check if the anchor box is already assigned to another annotation. numpy indexes rows (y) first, then columns (x)
    if target_tensor[grid_y, grid_x, anchor_idx, 4] == 1:

        # Compare IoU with the existing annotation
        # TODO could memoize for this image or cache
        existing_iou = calculate_iou(anchors[anchor_idx], target_tensor[grid_y, grid_x, anchor_idx, 2:4])
        current_iou = calculate_iou(anchors[anchor_idx], relative_bbox[2:4])

        # Assign the anchor box to the annotation with higher IoU
        if current_iou > existing_iou:
            target_tensor[grid_y, grid_x, anchor_idx, 5:] = 0  # Clear all class assignments
            write_to_tensor(anchor_idx, class_idx, grid_x, grid_y, relative_bbox, target_tensor)

    else:
        # Assign the anchor box to the current annotation
        write_to_tensor(anchor_idx, class_idx, grid_x, grid_y, relative_bbox, target_tensor)


def get_grid_cell_and_relative_bbox(bbox, grid_shape, image_height, image_width):
    """
    bbox is in the form of [x, y, w, h]
    If an object lies within multiple grid cells, split the bounding box.
    Get up to four new bboxes
    returns in the form of [((grid_y1, grid_x1), (bbox_cx1, bbox_cy1, bbox_w1, bbox_h1)), ...]
    """
    # TODO Don't I need to account for the padding added by the NN model here?
    cx, cy, w, h = bbox
    # TODO I don't need to calculate this every time, also see output
    grid_cell_shape = (image_height / grid_shape[0], image_width / grid_shape[1])
    grid_cell = int(cy / grid_cell_shape[0]), int(cx / grid_cell_shape[1])

    rel_cy = (cy - grid_cell[0] * grid_cell_shape[0]) / grid_cell_shape[0]
    rel_cx = (cx - grid_cell[1] * grid_cell_shape[1]) / grid_cell_shape[1]
    rel_height = h / grid_cell_shape[0]
    rel_width = w / grid_cell_shape[1]

    relative_bbox = [rel_cx, rel_cy, rel_width, rel_height]

    return grid_cell, relative_bbox


def get_best_anchor(bbox, anchors):
    """
    Get the anchor box with the highest IoU.

    Args:
        bbox (list): Bounding box coordinates.
        anchors (numpy.ndarray): Array of anchor boxes.

    Returns:
        int: Index of the anchor box with the highest IoU.
    """
    max_iou = -1
    best_anchor = -1
    for i, anchor in enumerate(anchors):
        iou = calculate_iou(bbox, anchor)
        if iou > max_iou:
            max_iou = iou
            best_anchor = i

    return best_anchor


def write_to_tensor(anchor_idx, class_idx, grid_x, grid_y, relative_bbox, target_tensor):
    target_tensor[grid_y, grid_x, anchor_idx, 0:4] = relative_bbox
    target_tensor[grid_y, grid_x, anchor_idx, 4] = 1  # objectness
    target_tensor[grid_y, grid_x, anchor_idx, 5 + class_idx] = 1  # assign new class


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
