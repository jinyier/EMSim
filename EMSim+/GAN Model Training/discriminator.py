from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras import Model




class encoder(Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = Conv2D(64, 3, activation='relu', padding='SAME')
        self.max1 = MaxPooling2D(2, padding='same')
        self.conv2 = Conv2D(32, 3, activation='relu', padding='SAME')
        self.max2 = MaxPooling2D(2, padding='same')
        self.conv3 = Conv2D(16, 5, activation='relu', padding='SAME')
        self.max3 = MaxPooling2D(2, padding='same')

    @tf.autograph.experimental.do_not_convert
    def call(self, vals, **kwargs):
        x = vals[0]
        y = vals[1]
        x = tf.concat((x, y), axis=-1)
        x0 = self.conv1(x)
        x1 = self.max1(x0)
        x1 = self.conv2(x1)
        x2 = self.max2(x1)
        x2 = self.conv3(x2)
        x3 = self.max3(x2)
        return x3


class ls_layer(Model):
    def __init__(self):
        super(ls_layer, self).__init__()
        self.fl = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', use_bias=True)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', use_bias=True)
        self.fc3 = tf.keras.layers.Dense(256, activation='relu', use_bias=True)
        self.fc4 = tf.keras.layers.Dense(576, activation='relu', use_bias=True)
        self.t_fc1 = tf.keras.layers.Dense(64, activation='relu', use_bias=True)
        self.t_fc2 = tf.keras.layers.Dense(64, activation='relu', use_bias=True)
        self.t_fc3 = tf.keras.layers.Dense(64, activation='relu', use_bias=True)
        self.conv4 = Conv2D(1, 3, activation='relu', padding='SAME')

    @tf.autograph.experimental.do_not_convert
    def call(self, vals, **kwargs):
        x = vals[0]
        t = vals[1]
        x = self.fl(x)
        x = self.fc1(x)
        x = self.fc2(x)
        t = self.t_fc1(t)
        t = self.t_fc2(t)
        t = self.t_fc3(t)
        x2 = Concatenate()([x, t])
        x2 = self.fc3(x2)
        x2 = self.fc4(x2)
        ls = tf.reshape(x2, [-1, 6, 6, 16])
        validity = self.conv4(ls)
        return validity


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ae = encoder()
        self.ls = ls_layer()

    @tf.autograph.experimental.do_not_convert
    def call(self, vals, **kwargs):
        x = vals[0]
        t = vals[1]
        y = vals[2]
        ae = self.ae((x, y))#6,6,16
        validity = self.ls((ae, t))
        return validity