"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import argparse
import logging
import datetime
from warmup import WarmUp
import tensorflow as tf
from embed_model import EmbedModel
from siamese_model import SiameseModel
from data_utils import make_dataset, feature_spec, train_files, eval_files
from loss import get_nnpu_loss_fn


def run() -> None:
    """Train PAUSE on SNLI and Multi-Genre NLI datasets with a certain label ratio."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="The tfhub link for the base embedding model",
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="The pretrained model if any",
    )
    parser.add_argument(
        "--train_epochs", default=4, type=int, help="The max number of training epoch"
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Training mini-batch size"
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
        "--prior",
        default=1.0 / 3.0,
        type=float,
        help="Expected ratio of positive samples",
    )
    parser.add_argument(
        "--train_lr", default=7.5e-05, type=float, help="The maximum learning rate"
    )
    parser.add_argument(
        "--pos_sample_prec",
        default="50",
        type=str,
        help="The percentage of sampled positive examples used in training; should be one of 1, 10, 30, 50, 70",
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
        train_files(opts.pos_sample_prec),
        opts.batch_size,
        "gold_label",
        opts.prior,
    )

    test_dataset = make_dataset(
        feature_spec,
        eval_files(opts.pos_sample_prec),
        opts.batch_size,
        "gold_label",
        opts.prior,
        False,
    )

    num_train_steps = opts.train_steps_per_epoch * opts.train_epochs
    num_warmup_steps = int(0.1 * num_train_steps)

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
        siamese_model = SiameseModel(
            bert_model,
        )

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
        lr_schedule = WarmUp(
            initial_learning_rate=opts.train_lr,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=opts.log_dir, update_freq="batch", profile_batch=0
        )
        nnpu_loss_weight = tf.Variable(0.0)
        siamese_model.compile(
            optimizer=optimizer,
            loss=get_nnpu_loss_fn(opts.prior, nnpu_loss_weight),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    class AnnealingCallback(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            nnpu_loss_weight.assign(pow(batch / num_train_steps, 3))

    siamese_model.fit(
        train_dataset.repeat(),
        epochs=opts.train_epochs,
        steps_per_epoch=opts.train_steps_per_epoch,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback, AnnealingCallback()],
    )

    siamese_model.save_model(opts.model_dir)

    bert_model.save_model(opts.model_dir)

    siamese_model.save_model(opts.model_dir, export_weights=True)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
