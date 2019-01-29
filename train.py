import os
import datetime

import cv2
import numpy as np
import tensorflow as tf

from anime import read_data_set
from networks import generator
from networks import discriminator
from config import Z_DIMENSIONS
from config import BATCH_SIZE
from config import ITERATION
from config import D_UPDATE
from config import G_UPDATE
from config import RESULT_ROOT


def main():
    # Load anime data set.
    anime = read_data_set()

    # The placehodler for feeding input noise to the generator.
    z_placeholder = tf.placeholder(tf.float32,
                                   [None, Z_DIMENSIONS],
                                   name='z_placeholder')
    # The placehodler for feeding input images to the discriminator.
    x_placeholder = tf.placeholder(tf.float32,
                                   shape=[None, 64, 64, 3],
                                   name='x_placeholder')

    # The generated images.
    g_z = generator(z_placeholder, BATCH_SIZE, Z_DIMENSIONS)
    # The discriminator prediction probability for the real images.
    d_x = discriminator(x_placeholder)
    # The discriminator prediction probability for the generated images.
    d_g = discriminator(g_z, reuse_variables=True)

    # Two Loss Functions for discriminator.
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_x,
        labels=tf.ones_like(d_x),
    ))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_g,
        labels=tf.zeros_like(d_g),
    ))
    # Loss function for generator.
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_g,
        labels = tf.ones_like(d_g),
    ))

    # Get the varaibles for different network.
    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    # Train the discriminator.
    d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake,
                                                             var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real,
                                                             var_list=d_vars)
    # Train the generator.
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

    # From this point forward, reuse variables.
    tf.get_variable_scope().reuse_variables()

    with tf.Session() as sess:
        # Send summary statistics to TensorBoard.
        tf.summary.scalar('Generator_loss', g_loss)
        tf.summary.scalar('Discriminator_loss_real', d_loss_real)
        tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

        folder_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        folder_root = os.path.join(RESULT_ROOT, folder_id)

        images_for_tensorboard = generator(z_placeholder, BATCH_SIZE, Z_DIMENSIONS)
        tf.summary.image('Generated_images', images_for_tensorboard, 5)
        merged = tf.summary.merge_all()
        logdir = os.path.join(folder_root, 'tensorboard')
        writer = tf.summary.FileWriter(logdir, sess.graph)

        model_dir = os.path.join(folder_root, 'model')
        model_path = '{}/model.ckpt'.format(model_dir)
        if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        img_dir = os.path.join(folder_root, 'images')
        if not os.path.exists(img_dir):
                os.makedirs(img_dir)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        # Pre-train discriminator.
        for i in range(300):
            z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMENSIONS])
            real_image_batch = anime.sample(BATCH_SIZE)
            _, __ = sess.run(
                [d_trainer_real, d_trainer_fake],
                {x_placeholder: real_image_batch, z_placeholder: z_batch},
            )

        # Train generator and discriminator together
        for i in range(ITERATION):
            real_image_batch = anime.sample(BATCH_SIZE)
            z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMENSIONS])

            # Train discriminator on both real and fake images.
            for j in range(D_UPDATE):
                _, __, d_loss_real_score, d_loss_fake_score = sess.run(
                    [d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                    {x_placeholder: real_image_batch, z_placeholder: z_batch},
                )

            # Train generator.
            for j in range(G_UPDATE):
                z_batch = np.random.normal(0,
                                           1,
                                           size=[BATCH_SIZE, Z_DIMENSIONS])
                _, g_loss_score = sess.run([g_trainer, g_loss],
                                           feed_dict={z_placeholder: z_batch})

            if i % 10 == 0:
                # Update TensorBoard with summary statistics.
                z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMENSIONS])
                summary = sess.run(
                    merged,
                    {z_placeholder: z_batch, x_placeholder: real_image_batch},
                )
                writer.add_summary(summary, i)

            if i % 100 == 0:
                z_batch = np.random.normal(0, 1, size=[1, Z_DIMENSIONS])
                generated_img = sess.run(g_z, { z_placeholder: z_batch })
                generated_img = generated_img.reshape([64, 64, 3])
                generated_img = generated_img * 128
                generated_img = generated_img + 127
                generated_img = generated_img.astype(np.uint8)
                cv2.imwrite('{}/{:05d}.png'.format(img_dir, i), generated_img)

            if i % 100 == 0:
                print("d_loss_real_score:", d_loss_real_score,
                      "d_loss_fake_score:", d_loss_fake_score,
                      "g_loss_score:", g_loss_score)

            saver.save(sess, model_path)


if __name__ == '__main__':
    main()
