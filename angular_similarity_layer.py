"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import math
import tensorflow as tf


class AngularSimilarity(tf.keras.layers.Layer):
    def __init__(self):
        super(AngularSimilarity, self).__init__()

    def call(self, x1, x2):
        cosine_sim = -tf.keras.losses.cosine_similarity(x1, x2)
        cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)
        cosine_sim = tf.expand_dims(cosine_sim, axis=-1)
        ang_distance = tf.math.acos(cosine_sim) / math.pi
        return 1.0 - ang_distance