from abc import ABC
from abc import abstractmethod

import tensorflow as tf


class Loss(ABC):

    def __init__(self, d_x, d_g):
        self._d_x = d_x
        self._d_g = d_g

    @abstractmethod
    def d_loss_real(self):
        pass

    @abstractmethod
    def d_loss_fake(self):
        pass

    @abstractmethod
    def g_loss(self):
        pass


class BasicLoss(Loss):

    def d_loss_real(self):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._d_x,
            labels=tf.ones_like(self._d_x),
        ))

    def d_loss_fake(self):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._d_g,
            labels=tf.zeros_like(self._d_g),
        ))

    def g_loss(self):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._d_g,
            labels = tf.ones_like(self._d_g),
        ))


class LeastSquaresLoss(Loss):

    def __init__(self, d_x, d_g, a, b, c):
        super().__init__(d_x, d_g)

        self._a = a
        self._b = b
        self._c = c

    def d_loss_real(self):
        return tf.reduce_mean(tf.square(self._d_x - self._b))

    def d_loss_fake(self):
        return tf.reduce_mean(tf.square(self._d_g - self._a))

    def g_loss(self):
        return tf.reduce_mean(tf.square(self._d_g - self._c))
