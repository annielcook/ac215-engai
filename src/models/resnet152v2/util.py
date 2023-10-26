import tensorflow as tf

def parse_tfrecord_example(example_proto, num_channels, num_classes, image_height, image_width):
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


def get_data(tfrecord_files, *, batch_size, num_channels, num_classes, image_height, image_width):
    data = tfrecord_files.flat_map(tf.data.TFRecordDataset)
    data = data.map(lambda x: parse_tfrecord_example(x, num_channels, num_classes, image_height, image_width), num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size)
    data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
    return data