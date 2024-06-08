# cnn_toolkit
Convolutional Neural Network Toolkit

## File list and purpose
1. run.py

Use this file to create a CNN from a dataset of annotated images.

python run.py --data_dir "path/to/your/data" --input_shape 1080 1920 3 --num_anchors 3 --batch_size 1 --epochs 30

2. run_test.py

Developers use this file to create a CNN and generate an indeifinte amount of test data for it.

python run.py --input_shape 200 200 3 --num_anchors 2 --num_samples 30 --batch_size 2 --epochs 40000

3. gpu_test.py

prints true if tensorflow is using the gpu.

python gpu_test.py

4. image_toolkit

useful functions for displaying your model output and for annotating images.
