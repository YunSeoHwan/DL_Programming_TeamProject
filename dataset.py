# dataset.py
# 데이터셋 준비 모듈

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image_and_labels(image_path, label, image_shape):
    image = img_to_array(load_img(image_path, target_size=image_shape[:2]))
    return image, label

def create_tf_dataset(image_paths, labels, image_shape, batch_size=32):
    def generator():
        for img_path, lbl in zip(image_paths, labels):
            yield load_image_and_labels(img_path, lbl, image_shape)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(image_shape[0], image_shape[1], image_shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
    ))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
