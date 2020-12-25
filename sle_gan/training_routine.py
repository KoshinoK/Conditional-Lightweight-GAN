import shutil
from pathlib import Path

import tensorflow as tf

import sle_gan


# 条件ベクトルをimagesの後の引数で受け取る
@tf.function
def train_step(G, D, G_optimizer, D_optimizer, images, conditional_vectors, diff_augmenter_policies: str = None) -> tuple:
    batch_size = images.shape[0]

    # Images for the I_{part} reconstruction loss
    images_batch_center_crop_128 = sle_gan.center_crop_images(images, 128)

    # Images for the I reconstruction loss
    image_batch_128 = tf.image.resize(images, (128, 128))

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
        """
        create_input_noise data.pyで定義。条件ベクトルを付加するように修正する。
        sle_gan.create_latent_vectors(batch_size, conditional_vectors)へ変更。
        """
        noise_input = sle_gan.data.create_latent_vectors(batch_size, conditional_vectors)
        
        generated_images = G(noise_input, training=True)

        """
        sle_gan.diff_augment(images, policy=diff_augmenter_policies)の戻り値に条件画像を付加する。
            -> diff_augmentの修正ではなく、この箇所で条件画像を付加する。
        """
        aug_real_images = sle_gan.diff_augment(images, policy=diff_augmenter_policies)
        dis_real_inputs = sle_gan.data.create_discriminator_inputs(aug_real_images, conditional_vectors)
        real_fake_output_logits_on_real_images, decoded_real_image, decoded_real_image_central_crop = D(
            dis_real_inputs, training=True)
        """
        sle_gan.diff_augment(generated_images, policy=diff_augmenter_policies)の戻り値に条件画像を付加する。
            -> diff_augmentの修正ではなく、この箇所で条件画像を付加する。
        """
        aug_fake_images = sle_gan.diff_augment(generated_images, policy=diff_augmenter_policies)
        dis_fake_inputs = sle_gan.data.create_discriminator_inputs(aug_fake_images, conditional_vectors)
        real_fake_output_logits_on_fake_images, _, _ = D(
            dis_fake_inputs, training=True)

        # Discriminator loss
        D_real_fake_loss = sle_gan.discriminator_real_fake_loss(
            real_fake_output_logits_on_real_images=real_fake_output_logits_on_real_images,
            real_fake_output_logits_on_fake_images=real_fake_output_logits_on_fake_images)
        D_I_reconstruction_loss = sle_gan.discriminator_reconstruction_loss(real_image=image_batch_128,
                                                                            decoded_image=decoded_real_image)
        D_I_part_reconstruction_loss = sle_gan.discriminator_reconstruction_loss(
            real_image=images_batch_center_crop_128,
            decoded_image=decoded_real_image_central_crop)
        D_loss = D_real_fake_loss + D_I_reconstruction_loss + D_I_part_reconstruction_loss

        # Generator loss
        G_loss = sle_gan.generator_loss(real_fake_output_logits_on_fake_images=real_fake_output_logits_on_fake_images)

    G_gradients = tape_G.gradient(G_loss, G.trainable_variables)
    D_gradients = tape_D.gradient(D_loss, D.trainable_variables)

    D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))
    G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))

    return G_loss, D_loss, D_real_fake_loss, D_I_reconstruction_loss, D_I_part_reconstruction_loss


def evaluation_step(inception_model:tf.keras.models.Model,
                    dataset: tf.keras.utils.Sequence,
                    G: tf.keras.models.Model,
                    batch_size: int,
                    image_height: int,
                    image_width: int,
                    nb_of_images_to_use: int = 128) -> float:
    number_of_batches = nb_of_images_to_use // batch_size

    # datasetから取得した条件ベクトルを格納するリスト
    conditional_vectors = []
    
    # Write real images from the dataset to the disk
    real_paths = []
    """
    条件ベクトルも呼び出すように修正する。
        -> 修正済み
    """
    for i in range(number_of_batches):
        x, v = dataset.__getitem__(i)
        conditional_vectors.append(v)
        real_images = sle_gan.postprocess_images(x, dtype=tf.uint8).numpy()
        _, real_images_file_paths = sle_gan.write_images_to_disk(real_images, folder=None)
        real_paths.extend(real_images_file_paths)
    """
    for x in dataset.take(number_of_batches):
        real_images = sle_gan.postprocess_images(x, dtype=tf.uint8).numpy()
        _, real_images_file_paths = sle_gan.write_images_to_disk(real_images, folder=None)
        real_paths.extend(real_images_file_paths)
    """
    
    # Generate images and write to the disk
    fake_paths = []
    for i in range(number_of_batches):
        """
        create_input_noise data.pyで定義。条件ベクトルを付加するように修正する。
        sle_gan.create_latent_vectors(batch_size, conditional_vectors)へ変更。
            -> 変更済み
        """
        input_noise = sle_gan.data.create_latent_vectors(batch_size, conditional_vectors[i])
        fake_images = G(input_noise)
        fake_images = sle_gan.postprocess_images(fake_images, dtype=tf.uint8).numpy()
        _, fake_images_file_paths = sle_gan.write_images_to_disk(fake_images, folder=None)
        fake_paths.extend(fake_images_file_paths)

    fid_score = sle_gan.calculate_FID(inception_model,
                                      real_paths,
                                      fake_paths,
                                      batch_size=batch_size,
                                      image_height=image_height,
                                      image_width=image_width)

    # Cleanup, remove the folders with all the written files
    shutil.rmtree(Path(real_paths[0]).parent)
    shutil.rmtree(Path(fake_paths[0]).parent)

    return fid_score
