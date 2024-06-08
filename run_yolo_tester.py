import json

from yolo_model import create_yolo_model, plot_loss
from yolo_data_generator import YoloDataGenerator
import numpy as np
from image_toolkit import overlay_output
import argparse


def main_test(input_shape, num_anchors, num_samples, batch_size=1, epochs=1):
    labels = ['circle', 'square']
    num_classes = len(labels)

    # Create the yolo model
    model = create_yolo_model(input_shape, num_classes, num_anchors)

    # Resulting grid
    grid_shape = model.output_shape[1:3]  # first dimension is batch size

    # Generate images with squares and circles scattered randomly
    train_generator = YoloDataGenerator(input_shape, batch_size, grid_shape, num_anchors, num_samples)
    val_generator = YoloDataGenerator(input_shape, batch_size, grid_shape, num_anchors, max(2, int(num_samples * 0.3)))

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
    )

    plot_loss(history)

    for images, true_ys in train_generator:
        for i in range(images.shape[0]):
            image = images[i]
            true_y = true_ys[i]
            output = model.predict(np.array([image]))
            output = output[0]  # no batch
            overlay_output(image, true_y, output, labels, objectness_threshold=0.15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Model Testing Script')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--input_shape', type=int, nargs=3, default=(500, 500, 3), help='Input shape of the model (height, width, channels)')
    parser.add_argument('--num_anchors', type=int, default=2, help='Number of anchor boxes')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of samples to generate per epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        input_shape = tuple(config['input_shape'])
        num_anchors = config['num_anchors']
        num_samples = config['num_samples']
        batch_size = config['batch_size']
        epochs = config['epochs']
    else:
        # Use command-line arguments if provided, otherwise use default values
        input_shape = tuple(args.input_shape)
        num_anchors = args.num_anchors
        num_samples = args.num_samples
        batch_size = args.batch_size
        epochs = args.epochs

    main_test(
        input_shape=input_shape,
        num_anchors=num_anchors,
        num_samples=num_samples,
        batch_size=batch_size,
        epochs=epochs
    )
