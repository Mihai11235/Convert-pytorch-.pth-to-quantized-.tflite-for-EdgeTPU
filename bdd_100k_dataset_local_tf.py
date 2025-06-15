import os
import tensorflow as tf
import numpy as np
from PIL import Image

def load_image_and_mask_1(img_path, mask_path, target_size=(128, 128)):
    # Load and process image
    if isinstance(target_size, tf.Tensor):
        target_size = tuple(target_size.numpy().tolist())
    image = Image.open(img_path.numpy().decode("utf-8")).convert("RGB").resize(target_size)
    mask = Image.open(mask_path.numpy().decode("utf-8")).convert("L").resize(target_size)


    image = np.array(image).astype(np.float32) / 255.0  # normalize [0, 1]
    mask = 1.0 - (np.array(mask).astype(np.float32) / 255.0)  # invert
    mask = (mask > 0.5).astype(np.float32)  # binarize
    mask = np.expand_dims(mask, axis=-1)  # make (H, W, 1)

    return image, mask


def load_bdd100k_dataset_tf_1(images_dir, masks_dir, target_size=(128, 128), batch_size=8, shuffle=True):
    image_filenames = sorted([
        os.path.join(images_dir, fname)
        for fname in os.listdir(images_dir) if fname.endswith('.jpg')
    ])

    mask_filenames = [
        os.path.join(masks_dir, os.path.basename(f).replace(".jpg", ".png"))
        for f in image_filenames
    ]

    image_filenames = np.array(image_filenames, dtype=str)
    mask_filenames = np.array(mask_filenames, dtype=str)
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))

    def process_path(img_path, mask_path):
        image, mask = tf.py_function(
            func=load_image_and_mask_1,
            inp=[img_path, mask_path, target_size],
            Tout=(tf.float32, tf.float32)
        )
        image.set_shape((target_size[0], target_size[1], 3))
        mask.set_shape((target_size[0], target_size[1], 1))
        return image, mask

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
