from yolo_model import create_yolo_model
from utilities import create_anchor_boxes, load_dataset
from yolo_data_converter import YoloDataConverter
import argparse
import json
from image_toolkit import overlay_output
import numpy as np
from sklearn.model_selection import train_test_split


def main(data_dir, input_shape, num_anchors, batch_size=1, epochs=1):

    image_resize = input_shape[0:2][::-1]  # h x w  ->  w x h

    # handles resizing, normalizing, and formatting bboxes to center coordinates + width/height
    images, annotations, labels = load_dataset(data_dir=data_dir,
                                               resize=image_resize,
                                               only_annotated=True,
                                               bbox_format='yolo')

    num_classes = len(labels)

    # Create the anchor boxes
    anchor_boxes = create_anchor_boxes(annotations, num_anchors)

    # Create the yolo model
    model = create_yolo_model(input_shape, num_classes, num_anchors)

    grid_shape = model.output_shape[1:3]  # first dimension is batch size

    train_images, val_images, train_annotations, val_annotations = train_test_split(
        images, annotations, test_size=0.3, random_state=42
    )

    # convert saved images and annotations to tensors
    train_generator = YoloDataConverter(train_images, train_annotations, batch_size, grid_shape,
                                        num_anchors, labels, anchor_boxes, shuffle=True, augment=None)

    validation_generator = YoloDataConverter(val_images, val_annotations, batch_size, grid_shape,
                                             num_anchors, labels, anchor_boxes, shuffle=False, augment=None)

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )

    for image in images:
        output = model.predict(np.array([image]))
        overlay_output(image, output, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Model Training Script')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--input_shape', type=int, nargs=3, default=(216, 384, 3), help='Input shape of the model (height, width, channels)')
    parser.add_argument('--num_anchors', type=int, default=2, help='Number of anchor boxes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        data_dir = config['data_dir']
        input_shape = tuple(config['input_shape'])
        num_anchors = config['num_anchors']
        batch_size = config['batch_size']
        epochs = config['epochs']
    else:
        # Use command-line arguments if provided, otherwise use default values
        data_dir = args.data_dir
        input_shape = tuple(args.input_shape)
        num_anchors = args.num_anchors
        batch_size = args.batch_size
        epochs = args.epochs

    main(
        data_dir=data_dir,
        input_shape=input_shape,
        num_anchors=num_anchors,
        batch_size=batch_size,
        epochs=epochs,
    )


# TODO
#  Try some other loss for w, h
#  Try Normalizing width & height
#  Make performance logging
#  Then figure out a smart way to get a good NN structure that works for yolo and automate it.
