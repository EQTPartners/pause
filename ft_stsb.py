"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import os
import argparse
import logging
import datetime
from typing import Any, Union
import tensorflow as tf
from embed_model import EmbedModel
from siamese_model import SiameseModel

# Feature specification (dictionary) of the pre-processed dataset
feature_spec = {
    "score": tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=None),
    "match_sentence": tf.io.FixedLenFeature(
        shape=[1], dtype=tf.string, default_value=None
    ),
    "sentence": tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=None),
    "uuid": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=None),
}


def make_dataset(
    feature_spec: dict,
    file_pattern: list,
    batch_size: int,
    label_key: str,
    training: bool = True,
) -> tf.data.Dataset:
    """Construct a train/eval dataset for funtuning PAUSE.

    Args:
        feature_spec (dict): The feature specification.
        file_pattern (list): The input TFRecord file patterns.
        batch_size (int): The training/evaluation batch size.
        label_key (str): The key of the label.
        training (bool, optional): Indicate if this is a training dataset. Defaults to True.

    Returns:
        tf.data.Dataset: The constructed dataset
    """

    def _parse_function(example_proto: Any) -> Union[dict, tf.Tensor]:
        """Parse feature and label from input example.

        Args:
            example_proto (Any): The input example (a scalar string Tensor).

        Returns:
            Union[dict, tf.Tensor]: The parsed feature and label.
        """

        _features = tf.io.parse_single_example(example_proto, feature_spec)
        _label = _features.pop(label_key)
        return _features, _label

    if training:
        dataset = tf.data.TFRecordDataset(
            filenames=tf.data.Dataset.list_files(file_pattern),
            compression_type="GZIP",
        )
        dataset = dataset.shuffle(200000)
    else:
        dataset = tf.data.TFRecordDataset(
            filenames=tf.data.Dataset.list_files(file_pattern),
            compression_type="GZIP",
        )

    dataset = dataset.map(
        _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def run()->None:
    """Finetune PAUSE on supervised STSb."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="The tfhub link for the base embedding model that should match pretrained model",
    )
    parser.add_argument(
        "--pretrained_weights",
        default="gs://motherbrain-pause/model/20210414-162525/serving_model_dir",
        type=str,
        help="The pretrained model if any",
    )
    parser.add_argument(
        "--train_epochs", default=4, type=int, help="The max number of training epoch"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Training mini-batch size"
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        default=500,
        type=int,
        help="Step interval of evaluation during training",
    )
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int,
        help="The max number of tokens in the input",
    )
    parser.add_argument(
        "--train_lr", default=7.5e-05, type=float, help="The maximum learning rate"
    )
    dt_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "--log_dir",
        default="./artifacts/log/{}".format(dt_str),
        type=str,
        help="The path where the logs are stored",
    )
    parser.add_argument(
        "--model_dir",
        default="./artifacts/model/{}".format(dt_str),
        type=str,
        help="The path where models and weights are stored",
    )

    opts, _ = parser.parse_known_args()
    print(opts)

    train_dataset = make_dataset(
        feature_spec,
        ["gs://motherbrain-pause/data/stsb/train/*"],
        opts.batch_size,
        "score",
    )

    test_dataset = make_dataset(
        feature_spec,
        ["gs://motherbrain-pause/data/stsb/test/*"],
        opts.batch_size,
        "score",
        False,
    )

    num_train_steps = opts.train_steps_per_epoch * opts.train_epochs

    bert_model_link = (
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"
    )
    if opts.model == "base":
        bert_model_link = (
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
        )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        bert_model = EmbedModel(
            bert_model_link,
            opts.max_seq_len,
        )
        siamese_model = SiameseModel(bert_model, is_reg=True)

        if opts.pretrained_weights != "":
            print("pretrained_weights_path=", opts.pretrained_weights)
            siamese_model.load_weights(
                os.path.join(opts.pretrained_weights, "saved_weights")
            )
            print("pre-trained model loaded!")

        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=opts.train_lr,
            decay_steps=num_train_steps,
            end_learning_rate=5e-6,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=opts.log_dir, update_freq="batch", profile_batch=0
        )
        siamese_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MSE,
            metrics=[tf.keras.metrics.MeanSquaredError()],
        )

    # Start training
    siamese_model.fit(
        train_dataset.repeat(),
        epochs=opts.train_epochs,
        steps_per_epoch=opts.train_steps_per_epoch,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback],
    )

    # Save Siamese Model
    siamese_model.save_model(opts.model_dir)

    # Save BERT model
    bert_model.save_model(opts.model_dir)

    # Save Model weights
    siamese_model.save_model(opts.model_dir, export_weights=True)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
