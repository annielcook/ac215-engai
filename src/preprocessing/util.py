import numpy as np
import tensorflow as tf

from PIL import Image


def to_tensor(image):
    # Convert image dtype from uint8 to float32 and scale values from [0, 255] to [0, 1]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Transpose the image from (H, W, C) to (C, H, W)
    # image = tf.transpose(image, (2, 0, 1))
    return image

def resize_img(blb, proc_bkt, curr_ext):
    local_image_file = 'curr_image' + curr_ext
    blb.download_to_filename(local_image_file)
    image = Image.open(local_image_file)
    image_tensor = to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensors = tf.image.resize(image_tensor, [224, 224], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
    img = tf.image.convert_image_dtype(image_tensors[0], dtype=tf.uint8)
    Image.fromarray(np.array(img)).save(local_image_file)
    destination_blob = proc_bkt.blob(blb.name)
    destination_blob.upload_from_filename(local_image_file)