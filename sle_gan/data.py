from functools import partial

import tensorflow as tf
import numpy as np

def create_input_noise(batch_size: int):
    return tf.random.normal(shape=(batch_size, 1, 1, 256), mean=0.0, stddev=1.0, dtype=tf.float32)

def create_latent_vectors(batch_size: int, conditional_vectors):
    """
    ノイズベクトルと条件ベクトルを結合したベクトルを生成する。
    
    Args:
        batch_size: バッチサイズ
        conditional_vectors: 条件ベクトル (B, NUM_CLASSES)
    
    Returns
        latent_vectors: (B, 1, 1, 256 + NUM_CLASSES)
    """
    noise = np.random.randn(batch_size, 256).astype('float32')
#     print('type of noise = ', type(noise))
#     print('type of conditional_vectors = ', type(conditional_vectors))    
    return tf.concat([noise, conditional_vectors], axis=1)
#     return np.concatenate([noise, conditional_vectors], axis=1)

def create_discriminator_inputs(images, conditional_vectors):
    """
    識別器用入力画像（画像＋条件画像）を生成する。
    Args:
        images: 画像
        conditional_vectors: 条件ベクトル
        index: image_seqから取得するデータのインデックス
    
    Returns
        画像＋条件画像を統合したテンソル (B, H, W, A + C)
        B: バッチサイズ。images.shape[0]
        H: 画像の高さ。images.shape[1]
        W: 画像の幅。images.shape[2]
        A: 画像の成分数。images.shape[3]
        C: 条件ベクトルの次元
    """
    
    # eagerモードにしておかないと、tensor.numpy()やスライスが使用困難
    tf.config.experimental_run_functions_eagerly(True)
    
    conditional_images = np.zeros((images.shape[0], images.shape[1], images.shape[2], conditional_vectors.shape[-1]),
                                    dtype='float32')
        
    if tf.is_tensor(conditional_vectors):
        conditional_vectors = conditional_vectors.numpy()
        conditional_images[:, ] = conditional_vectors.reshape((images.shape[0], 1, 1, conditional_vectors.shape[-1]))
    else:
        conditional_images[:, ] = conditional_vectors.reshape((images.shape[0], 1, 1, conditional_vectors.shape[-1]))
    return tf.concat([images, conditional_images], axis=-1)
#     return np.concatenate([images, conditional_images], axis=-1)
    
def read_image_from_path(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return image


def preprocess_images(images, resolution: int):
    """
    Resize and normalize the images tot he range [-1, 1]
    Args:
        images: batch of images (B, H, W, C)

    Returns:
        resized and normalized images
    """

    images = tf.image.resize(images, (resolution, resolution))
    images = tf.cast(images, tf.float32) - 127.5
    images = images / 127.5
    return images


def postprocess_images(images, dtype=tf.float32):
    """
    De-Normalize the images to the range [0, 255]
    Args:
        images: batch of normalized images
        dtype: target dtype

    Returns:
        de-normalized images
    """

    images = (images * 127.5) + 127.5
    images = tf.cast(images, dtype)
    return images


def create_dataset(batch_size: int,
                   folder: str,
                   resolution: int,
                   use_flip_augmentation: bool = True,
                   image_extension: str = "jpg",
                   shuffle_buffer_size: int = 100):
    dataset = tf.data.Dataset.list_files(folder + f"/*.{image_extension}")
    dataset = dataset.map(read_image_from_path)
    if use_flip_augmentation:
        dataset = dataset.map(tf.image.flip_left_right)
    dataset = dataset.map(partial(preprocess_images, resolution=resolution))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def center_crop_images(images, crop_resolution: int):
    """
    Crops the center of the images
    Args:
        images: shape: (B, H, W, 3), H should be equal to W
        crop_resolution: target resolution for the crop

    Returns:
        cropped images which has the shape: (B, crop_resolution, crop_resolution, 3)
    """

    # crop_resolution = tf.cast(crop_resolution, tf.float32)
    half_of_crop_resolution = crop_resolution / 2
    image_height = images.shape[1]
    image_center = image_height / 2

    from_ = int(image_center - half_of_crop_resolution)
    to_ = int(image_center + half_of_crop_resolution)

    return images[:, from_:to_, from_:to_, :]


def get_test_images(batch_size: int, folder: str, resolution: int):
    dataset = create_dataset(batch_size, str(folder), resolution=resolution, use_flip_augmentation=False,
                             shuffle_buffer_size=1)
    for x in dataset.take(1):
        return x

# def get_discriminator_input(image_seq, batch_size, index):
#     images, conditional_vectors = image_seq.__getitem__(index)
#     num_classes = conditional_vectors.shape[-1]
#     conditional_images = np.zeros((batch_size, input_resolution, input_resolution, 
#                                     conditional_vectors.shape[-1]), dtype='float32')
#     conditional_images[:, ] = conditional_vectors.reshape(batch_size, 1, 1, num_classes)
#     return np.concatenate([images, conditional_images], axis=-1)

