"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

_FEATURE_KEY = "sentence"


class EmbedModel(tf.keras.Model):
    def __init__(
        self,
        bert_model_link,
        max_seq_len=128,
    ):
        super(EmbedModel, self).__init__()
        print("bert_model_link=", bert_model_link)
        self.bert_layer = hub.KerasLayer(bert_model_link, trainable=True)

        self._sep_id = tf.constant(102, dtype=tf.int32)
        self._cls_id = tf.constant(101, dtype=tf.int32)
        self._pad_id = tf.constant(0, dtype=tf.int32)
        self.max_seq_len = max_seq_len
        self.tokenizer = text.BertTokenizer(
            self.bert_layer.resolved_object.vocab_file.asset_path,
            lower_case=True,
            token_out_type=tf.int32,
        )

    def tokenize_single_sentence_unpad(self, sequence: tf.Tensor):
        word_ids = self.tokenizer.tokenize(sequence)
        word_ids = word_ids.merge_dims(-2, -1)
        cls_token = tf.fill([word_ids.nrows(), 1], self._cls_id)
        word_ids = tf.concat([cls_token, word_ids], 1)

        sep_token = tf.fill([word_ids.nrows(), 1], self._sep_id)
        word_ids = word_ids[:, : self.max_seq_len - 1]
        word_ids = tf.concat([word_ids, sep_token], 1)

        return word_ids

    def tokenize_single_sentence_pad(self, sequence: tf.Tensor):
        word_ids = self.tokenize_single_sentence_unpad(
            tf.reshape(sequence, [-1])
        )
        word_ids = word_ids.to_tensor(
            shape=[None, self.max_seq_len], default_value=self._pad_id,
        )
        input_mask = tf.cast(tf.not_equal(word_ids, self._pad_id), tf.int32)
        segment_ids = tf.zeros_like(word_ids, tf.int32)
        return {
            "input_word_ids": word_ids,
            "input_mask": input_mask,
            "input_type_ids": segment_ids,
        }

    def call(self, data):
        bert_input = self.tokenize_single_sentence_pad(data)
        bert_out = self.bert_layer(bert_input)
        bert_out_selected = tf.math.reduce_mean(bert_out["sequence_output"], axis=1)
        return bert_out_selected

    def save_model(self, filepath):
        serving_model_split = filepath.split("/")
        model_dir = "/".join(
            serving_model_split[: len(serving_model_split) - 1]
        )
        embed_model_dir = os.path.join(
            model_dir, "embed", serving_model_split[-1]
        )
        tf.get_logger().info(
            "Saving embed model to {}".format(embed_model_dir)
        )
        signatures = {
            "serving_default": self._get_serve_tf_examples_fn().get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
            ),
        }
        self.save(
            filepath=embed_model_dir,
            overwrite=True,
            include_optimizer=False,
            save_format="tf",
            signatures=signatures,
        )

    def _get_serve_tf_examples_fn(self):
        """Returns a function that parses a serialized tf.Example."""

        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples):
            """Returns the output to be used in the serving signature."""
            feature_spec = {
                _FEATURE_KEY: tf.io.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.io.parse_example(
                serialized_tf_examples, feature_spec
            )
            output = self(parsed_features[_FEATURE_KEY])
            return {"output": output}

        return serve_tf_examples_fn
