import tensorflow as tf


def weights_initializer(stddev=0.02):
    return tf.truncated_normal_initializer(stddev=stddev)


def discriminator(images, ranged_output, reuse_variables=None):
     with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # Convolution to 32 x 32 x 32.
        d_w1 = tf.get_variable('d_w1',
                               [4, 4, 3, 32],
                               initializer=weights_initializer())
        d_b1 = tf.get_variable('d_b1',
                               [32],
                               initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images,
                          filter=d_w1,
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.contrib.layers.batch_norm(d1, epsilon=1e-5, scope='bn1_d')
        d1 = tf.nn.leaky_relu(d1)

        # Convolution to 16 x 16 x 64.
        d_w2 = tf.get_variable('d_w2',
                               [4, 4, 32, 64],
                               initializer=weights_initializer())
        d_b2 = tf.get_variable('d_b2',
                               [64],
                               initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1,
                          filter=d_w2,
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.contrib.layers.batch_norm(d2, epsilon=1e-5, scope='bn2_d')
        d2 = tf.nn.leaky_relu(d2)

        # Convolution to 8 x 8 x 128.
        d_w3 = tf.get_variable('d_w3',
                               [4, 4, 64, 128],
                               initializer=weights_initializer())
        d_b3 = tf.get_variable('d_b3',
                               [128],
                               initializer=tf.constant_initializer(0))
        d3 = tf.nn.conv2d(input=d2,
                          filter=d_w3,
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        d3 = d3 + d_b3
        d3 = tf.contrib.layers.batch_norm(d3, epsilon=1e-5, scope='bn3_d')
        d3 = tf.nn.leaky_relu(d3)

        # Convolution to 4 x 4 x 256.
        d_w4 = tf.get_variable('d_w4',
                               [4, 4, 128, 256],
                               initializer=weights_initializer())
        d_b4 = tf.get_variable('d_b4',
                               [256],
                               initializer=tf.constant_initializer(0))
        d4 = tf.nn.conv2d(input=d3,
                          filter=d_w4,
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        d4 = d4 + d_b4
        d4 = tf.contrib.layers.batch_norm(d4, epsilon=1e-5, scope='bn4_d')
        d4 = tf.nn.leaky_relu(d4)

        # Fully connected layer, output a scalar.
        d_w5 = tf.get_variable('d_w5',
                               [4 * 4 * 256, 1],
                               initializer=weights_initializer())
        d_b5 = tf.get_variable('d_b5',
                               [1],
                               initializer=tf.constant_initializer(0))
        d5 = tf.reshape(d4, [-1, 4 * 4 * 256])
        d5 = tf.matmul(d5, d_w5)
        d5 = d5 + d_b5

        if ranged_output:
            g5 = tf.sigmoid(d5)

        return d5


def generator(z, batch_size, z_dim):
    # From z_dim to 16 x 16 x 128 dimension.
    g_w1 = tf.get_variable('g_w1',
                           [z_dim, 16 * 16 * 128],
                           dtype=tf.float32,
                           initializer=weights_initializer())
    g_b1 = tf.get_variable('g_b1',
                           [16 * 16 * 128],
                           initializer=weights_initializer())
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 16, 16, 128])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1_g')
    g1 = tf.nn.leaky_relu(g1)
    # Upsampling to 32 x 32 x 128 dimension.
    g1 = tf.image.resize_images(g1, [32, 32])

    # Convolution to 32 x 32 x 128.
    g_w2 = tf.get_variable('g_w2',
                           [4, 4, 128, 128],
                           dtype=tf.float32,
                           initializer=weights_initializer())
    g_b2 = tf.get_variable('g_b2',
                           [128],
                           initializer=weights_initializer())
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 1, 1, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2_g')
    g2 = tf.nn.leaky_relu(g2)
    # Upsampling to 64 x 64 x 128 dimension.
    g2 = tf.image.resize_images(g1, [64, 64])

    # Convolution to 64 x 64 x 64.
    g_w3 = tf.get_variable('g_w3',
                           [4, 4, 128, 64],
                           dtype=tf.float32,
                           initializer=weights_initializer())
    g_b3 = tf.get_variable('g_b3',
                           [64],
                           initializer=weights_initializer())
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 1, 1, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3_g')
    g3 = tf.nn.leaky_relu(g3)

    # Convolution to 64 x 64 x 3.
    g_w4 = tf.get_variable('g_w4',
                           [4, 4, 64, 3],
                           dtype=tf.float32,
                           initializer=weights_initializer())
    g_b4 = tf.get_variable('g_b4',
                           [3],
                           initializer=weights_initializer())
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 1, 1, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.tanh(g4)

    # Dimensions of g4: batch_size x 64 x 64 x 3
    return g4
