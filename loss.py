"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


from typing import Callable
import tensorflow as tf


def calculate_losses(
    network_out: tf.Tensor, labels: tf.Tensor, label_idx: int, prior: float
) -> tf.Tensor:
    """Calculate the loss value for a certain class.

    Args:
        network_out (tf.Tensor): The output of the siamese network.
        labels (tf.Tensor): The ground truth label.
        label_idx (int): The class ID to calculate loss for.
        prior (float): The prior of positive ratio.

    Returns:
        tf.Tensor: The calculated loss value for class label_idx.
    """

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


def get_nnpu_loss_fn(prior: float, nnpu_weight: tf.Tensor) -> Callable:
    """Obtain the loss function of PAUSE.

    Args:
        prior (float): The prior of positive rate.
        nnpu_weight (tf.Tensor): The current PU loss weight.

    Returns:
        function: The overall loss function.
    """

    def nnpu_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate the PAUSE loss value

        Args:
            y_true (tf.Tensor): The ground truth label.
            y_pred (tf.Tensor): The predicted output.

        Returns:
            tf.Tensor: The value of the loss.
        """

        _loss = 0.0

        # Change num_classes according to dataset spec.
        num_classes = 3

        # Calculate PU Loss
        for i in range(num_classes):
            _loss += calculate_losses(y_pred[:, i : i + 1], y_true, i, prior)
        _loss *= nnpu_weight * (1 / num_classes)

        # Calculate Cross Entropy Loss
        y_labeled_mask = tf.greater(tf.reshape(y_true, [-1]), -1)
        y_labeled_idx = tf.reshape(tf.where(y_labeled_mask), [-1])
        y_true_gathered = tf.gather(y_true, y_labeled_idx, axis=0)
        y_pred_gathered = tf.gather(y_pred, y_labeled_idx, axis=0)
        _loss += tf.keras.losses.sparse_categorical_crossentropy(
            y_true_gathered, y_pred_gathered, from_logits=True
        )

        return _loss

    return nnpu_loss
