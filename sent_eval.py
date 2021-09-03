"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import argparse
import logging
import tensorflow as tf
import senteval
from tensorflow.core.example import example_pb2, feature_pb2


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    embeddings = params["predict_fn"](
        examples=tf.constant([make_example(sent) for sent in batch])
    )["output"].numpy()
    return embeddings


def make_example(text):
    ex = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                "sentence": feature_pb2.Feature(
                    bytes_list=feature_pb2.BytesList(value=[text.encode()])
                ),
            }
        )
    )
    return ex.SerializeToString()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to SentEval data",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The trained embed model",
    )
    args = parser.parse_args()

    model_dir = f"gs://motherbrain-pause/model/{args.model}/embed/serving_model_dir/"
    loaded_model = tf.saved_model.load(model_dir)
    predict_fn = loaded_model.signatures["serving_default"]

    # Set params for SentEval
    params_senteval = {
        "task_path": args.data_path,
        "usepytorch": True,
        "kfold": 10,
        "classifier": {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        },
        "predict_fn": predict_fn,
    }

    # Set up logger
    logging.basicConfig(
        format="%(asctime)s : %(message)s", level=logging.DEBUG
    )

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
        "MR",
        "CR",
        "MPQA",
        "SUBJ",
        "TREC",
        "MRPC",
    ]
    results = se.eval(transfer_tasks)
    print(args.model, results)
    with open(f"sent_eval_{args.model}.txt", "w") as out_file:
        out_file.write(results)
