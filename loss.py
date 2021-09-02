"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import tensorflow as tf


def calculate_losses(network_out, labels, label_idx, prior):
    assert network_out.shape.ndims == 2
    assert network_out.shape[1] == 1
    loss_func = lambda network_out, y: tf.nn.sigmoid(-network_out * y)
    positive = tf.cast(tf.equal(labels, label_idx), tf.float32)
    unlabeled = tf.cast(tf.equal(labels, -1), tf.float32)
    num_positive = tf.maximum(1.0, tf.reduce_sum(positive))
    num_unlabeled = tf.maximum(1.0, tf.reduce_sum(unlabeled))
    losses_positive = loss_func(network_out, 1)
    losses_negative = loss_func(network_out, -1)

    positive_risk = tf.reduce_sum(prior * positive / num_positive * losses_positive)
    negative_risk = tf.reduce_sum(
        (unlabeled / num_unlabeled - prior * positive / num_positive) * losses_negative,
    )
    return tf.cond(
        tf.less(negative_risk, -0.0),
        lambda: -1.0 * negative_risk,
        lambda: positive_risk + negative_risk,
    )


def get_nnpu_loss_fn(prior, nnpu_weight):
    def nnpu_loss(y_true, y_pred):
        _loss = 0.0
        # Calc nnPU loss
        for i in range(3):
            _loss += calculate_losses(y_pred[:, i : i + 1], y_true, i, prior)
        _loss *= nnpu_weight * (1 / 3)

        # Calc CE loss
        # print('y_true', y_true)
        y_labeled_mask = tf.greater(tf.reshape(y_true, [-1]), -1)
        # print('y_labeled_mask', y_labeled_mask)
        y_labeled_idx = tf.reshape(tf.where(y_labeled_mask), [-1])
        # print('y_labeled_idx', y_labeled_idx)
        y_true_gathered = tf.gather(y_true, y_labeled_idx, axis=0)
        # print('y_true_gathered', y_true_gathered)
        y_pred_gathered = tf.gather(y_pred, y_labeled_idx, axis=0)
        # print('y_pred_gathered', y_pred_gathered)
        _loss += tf.keras.losses.sparse_categorical_crossentropy(
            y_true_gathered, y_pred_gathered, from_logits=True
        )

        return _loss

    return nnpu_loss