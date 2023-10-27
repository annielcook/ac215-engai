import json
import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params
import matplotlib.pyplot as plt




def parse_tfrecord_example(
    example_proto, num_channels, num_classes, image_height, image_width
):
    # Create a dictionary with the image data and label
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channel": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Image
    image = tf.io.decode_raw(parsed_example["image"], tf.uint8)
    image.set_shape([num_channels * image_height * image_width])
    image = tf.reshape(image, [image_height, image_width, num_channels])

    # Label
    label = tf.cast(parsed_example["label"], tf.int64)
    label = tf.one_hot(label, num_classes)

    return image, label


# Normalize pixels
def normalize(image, label):
    image = image / 255
    return image, label


def get_data(
    tfrecord_files, *, batch_size, num_channels, num_classes, image_height, image_width
):
    data = tfrecord_files.flat_map(tf.data.TFRecordDataset)
    data = data.map(
        lambda x: parse_tfrecord_example(
            x, num_channels, num_classes, image_height, image_width
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    data = data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size)
    data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
    return data

class JsonEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return super(JsonEncoder, self).default(obj)

def get_model_size(model_name="model01", experiment_name="experiment01"):
  model_size = os.stat(os.path.join(experiment_name,model_name+".hdf5")).st_size
  return model_size

def save_model(
    experiment_name,
    model,
    model_train_history,
    batch_size,
    execution_time,
    learning_rate,
    epochs,
    optimizer,
    evaluation_results,
):
    model_name = model.name

    # Ensure path exists
    if not os.path.exists(experiment_name):
        os.mkdir(experiment_name)
    # Save the enitire model (structure + weights)
    model.save(os.path.join(experiment_name, model_name + ".hdf5"))

    # Save only the weights
    model.save_weights(os.path.join(experiment_name, model_name + ".h5"))

    # Save the structure only
    model_json = model.to_json()
    with open(os.path.join(experiment_name, model_name + ".json"), "w") as json_file:
        json_file.write(model_json)

    model_size = get_model_size(model_name=model.name, experiment_name=experiment_name)

    # Save model history
    with open(
        os.path.join(experiment_name, model.name + "_train_history.json"), "w"
    ) as json_file:
        json_file.write(json.dumps(model_train_history, cls=JsonEncoder))

    trainable_parameters = count_params(model.trainable_weights)
    non_trainable_parameters = count_params(model.non_trainable_weights)

    # Save model metrics
    metrics = {
        "trainable_parameters": trainable_parameters,
        "execution_time": execution_time,
        "loss": evaluation_results[0],
        "accuracy": evaluation_results[1],
        "model_size": model_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": type(optimizer).__name__,
    }
    with open(
        os.path.join(experiment_name, model.name + "_model_metrics.json"), "w"
    ) as json_file:
        json_file.write(json.dumps(metrics, cls=JsonEncoder))


def evaluate_model(
    model,
    test_data,
    model_train_history,
    execution_time,
    learning_rate,
    batch_size,
    epochs,
    optimizer,
    save=True,
    loss_metrics=["loss", "val_loss"],
    acc_metrics=["accuracy", "val_accuracy"],
    experiment_name="experiment1"
):
    # Get the number of epochs the training was run for
    num_epochs = len(model_train_history[loss_metrics[0]])

    # Plot training results
    fig = plt.figure(figsize=(15, 5))
    axs = fig.add_subplot(1, 2, 1)
    axs.set_title("Loss")
    # Plot all metrics
    for metric in loss_metrics:
        axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
    axs.legend()

    axs = fig.add_subplot(1, 2, 2)
    axs.set_title("Accuracy")
    # Plot all metrics
    for metric in acc_metrics:
        axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
    axs.legend()

    plt.show()

    # Evaluate on test data
    evaluation_results = model.evaluate(test_data, return_dict=True)
    print(evaluation_results)

    evaluation_results = [
        evaluation_results[loss_metrics[0]],
        evaluation_results[acc_metrics[0]],
    ]

    if save:
        # Save model
        save_model(
            experiment_name,
            model,
            model_train_history,
            batch_size,
            execution_time,
            learning_rate,
            epochs,
            optimizer,
            evaluation_results,
        )

    return evaluation_results
