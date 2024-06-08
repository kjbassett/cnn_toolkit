import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import datetime


def create_yolo_model(input_shape, num_classes, num_anchors):
    inputs = keras.Input(shape=input_shape)

    # Feature Extractor
    x = keras.layers.Conv2D(32, 15, padding='same', activation='relu')(inputs)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(64, 8, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(128, 5, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling2D(2)(x)

    # Detection Head
    # "fully connected" for each grid cell
    x = keras.layers.Conv2D(2048, 1, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(2048, 1, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(num_anchors * (num_classes + 5), 1)(x)

    # Reshape the output tensor
    output_shape = x.shape
    x = tf.reshape(x, shape=(-1, output_shape[1], output_shape[2], num_anchors, num_classes + 5))

    # Apply activation functions to the relevant parts of the tensor
    pred_xy = tf.sigmoid(x[..., 0:2])  # Sigmoid for x and y
    pred_wh = tf.exp(x[..., 2:4])  # Exponentiation for w and h
    pred_conf = tf.sigmoid(x[..., 4:5])  # Sigmoid for objectness score
    pred_class = tf.nn.softmax(x[..., 5:], axis=-1)  # Softmax for class predictions

    outputs = tf.concat([pred_xy, pred_wh, pred_conf, pred_class], axis=-1)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=yolo_loss)
    return model


def save_model(model):
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_shape = 'x'.join(model.input_shape)
    model.save(f'yolo_model_{dt}_{input_shape}.h5')


def yolo_loss(y_true, y_pred, lambda_coord=3.5):
    """
    Assuming y_true and y_pred are tensors with shape:
    batch_size x grid_size x grid_size x num_anchors x (num_classes + 5)
    where the last dimension contains (y, x, h, w, objectness, class_probs)
    :param y_true:
    :param y_pred:
    :return: total_loss
    """

    pred_xy = y_pred[..., 0:2]
    pred_wh = y_pred[..., 2:4]
    pred_conf = y_pred[..., 4]
    pred_class = y_pred[..., 5:]

    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    true_conf = y_true[..., 4]
    true_class = y_true[..., 5:]

    # Calculate the localization loss
    coord_loss = tf.reduce_sum(tf.square(pred_xy - true_xy) + tf.square(pred_wh - true_wh), axis=[-1])
    coord_loss = tf.reduce_sum(coord_loss * true_conf) * lambda_coord

    # Calculate the objectness loss
    obj_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(true_conf, pred_conf)

    # Calculate the classification loss
    class_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(true_class, pred_class)
    class_loss = tf.reduce_sum(class_loss * true_conf)

    # Combine the losses
    total_loss = coord_loss + obj_loss + class_loss

    return total_loss


def plot_loss(history):
    # Create traces
    trace1 = go.Scatter(
        y=history.history['loss'],
        mode='lines',
        name='Training Loss'
    )
    trace2 = go.Scatter(
        y=history.history['val_loss'],
        mode='lines',
        name='Validation Loss'
    )
    data = [trace1, trace2]
    # Layout can be adjusted as needed
    layout = go.Layout(
        title='Loss Over Time',
        xaxis=dict(title='Epochs'),
        yaxis=dict(title='Loss')
    )
    fig = go.Figure(data=data, layout=layout)
    # Show plot
    fig.show()


