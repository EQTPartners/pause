"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


from typing import Any, Union
import tensorflow as tf
import os
from angular_similarity_layer import AngularSimilarity
from embed_model import EmbedModel, _FEATURE_KEY

_MATCH_FEATURE_KEY = "match_sentence"


class SiameseModel(tf.keras.Model):
    """The implementation of the duel-encoder (siamese) model."""

    def __init__(self, bert_model: EmbedModel, is_reg: bool = False) -> None:
        """Initialize a duel-encoder model.

        Args:
            bert_model (EmbedModel): The embedding model instance.
            is_reg (bool, optional): Is using regression target? Defaults to False.
        """
        super(SiameseModel, self).__init__()
        self.output_names = ["logit", "acos_sim", "cos_sim"]
        self.bert_model = bert_model
        self.angular_similarity = AngularSimilarity()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.elu)
        self.dense2 = tf.keras.layers.Dense(3)
        self.is_reg = is_reg
        if is_reg:
            self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.relu)

    def call(self, data: dict) -> Union[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Implements the operations in siamese model.

        Args:
            data (dict): The input sentence pairs.

        Returns:
            Union[tf.Tensor, tf.Tensor, tf.Tensor]: Network output and calculated similarity scores.
        """
        bert_out_a = self.bert_model(data[_FEATURE_KEY])
        bert_out_b = self.bert_model(data[_MATCH_FEATURE_KEY])
        acos_sim = self.angular_similarity(bert_out_a, bert_out_b)
        diff = bert_out_a - bert_out_b
        abs_diff = tf.abs(diff)
        ele_prod = bert_out_a * bert_out_b
        concatenated = self.concat_layer([bert_out_a, bert_out_b, abs_diff, ele_prod])
        logit = self.dense1(concatenated)
        logit = self.dense2(logit)
        if self.is_reg:
            logit = self.dense3(logit)
        cos_sim = -tf.keras.losses.cosine_similarity(bert_out_a, bert_out_b)
        cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
        cos_sim = tf.expand_dims(cos_sim, axis=-1)
        return logit, acos_sim, cos_sim

    def train_step(self, data: Union[dict, tf.Tensor]) -> dict:
        """One training step.

        Args:
            data (Union[dict, tf.Tensor]): The input features and label.

        Returns:
            dict: he evaluation metrics on training batch.
        """
        _features, _label = data
        with tf.GradientTape() as tape:
            pred, _, _ = self(_features, training=True)
            loss = self.compiled_loss(
                _label,
                pred,
                regularization_losses=self.losses,
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(_label, pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Union[dict, tf.Tensor]) -> dict:
        """One test step.

        Args:
            data (Union[dict, tf.Tensor]): The input features and label.

        Returns:
            dict: The evaluation metrics on test batch.
        """
        _features, _label = data
        pred, _, _ = self(_features, training=False)
        self.compiled_loss(_label, pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(_label, pred)
        return {m.name: m.result() for m in self.metrics}

    def save_model(self, filepath: str, export_weights: bool = False) -> None:
        """Save the trained siamese model

        Args:
            filepath (str): The folder to which the model will be saved.
            export_weights (bool, optional): Also save weights snapshots when True. Defaults to False.
        """
        tf.get_logger().info("Saving model: {}".format(filepath))
        signatures = {
            "serving_default": self._get_serve_tf_examples_fn().get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
            ),
        }
        self.save(
            filepath=filepath,
            overwrite=True,
            include_optimizer=True,
            save_format="tf",
            signatures=signatures,
        )
        if export_weights:
            self.save_weights(os.path.join(filepath, "saved_weights"))

    def _get_serve_tf_examples_fn(self) -> tf.function:
        """Returns a function that parses a serialized tf.Example."""

        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples: tf.Tensor) -> dict:
            """Returns the output to be used in the serving signature."""
            feature_spec = {
                _FEATURE_KEY: tf.io.FixedLenFeature([], tf.string),
                _MATCH_FEATURE_KEY: tf.io.FixedLenFeature([], tf.string),
            }
            parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
            logit, acos_sim, cos_sim = self(parsed_features)
            return {
                self.output_names[0]: logit,
                self.output_names[1]: acos_sim,
                self.output_names[2]: cos_sim,
            }

        return serve_tf_examples_fn
