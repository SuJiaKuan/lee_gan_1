import argparse

import cv2
import numpy as np
import tensorflow as tf

from networks import generator


Z_DIMENSIONS = 100
BATCH_SIZE = 50


def main(model_dir):
    # The placehodler for feeding input noise to the generator.
    z_placeholder = tf.placeholder(tf.float32,
                                   [None, Z_DIMENSIONS],
                                   name='z_placeholder')

    # The generator.
    g_z = generator(z_placeholder, BATCH_SIZE, Z_DIMENSIONS)

    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)

        z_batch = np.random.normal(0, 1, size=[1, Z_DIMENSIONS])
        generated_img = sess.run(g_z, { z_placeholder: z_batch })[0]

        generated_img = generated_img * 128
        generated_img = generated_img + 127
        generated_img = generated_img.astype(np.uint8)

        cv2.imshow('Generated Image', generated_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference of generator')
    parser.add_argument('model_dir',
                        type=str,
                        help="Path to model directory")
    args = parser.parse_args()

    main(args.model_dir)
