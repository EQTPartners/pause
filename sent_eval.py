"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import os
import json
import argparse
import logging
import tensorflow as tf
import senteval
from tensorflow.core.example import example_pb2, feature_pb2

# The following import is mandatory
import tensorflow_text as text
import numpy as np


def prepare(params: senteval.utils.dotdict, samples: list) -> None:
    """Stub function required by SentEval"""
    return


def batcher(params: senteval.utils.dotdict, batch: list) -> np.ndarray:
    """Transforms a batch of text sentences into sentence embeddings.

    Args:
        params (senteval.utils.dotdict): [description]
        batch (list): [description]

    Returns:
        np.ndarray: [description]
    """
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    embeddings = params["predict_fn"](
        examples=tf.constant([make_example(sent) for sent in batch])
    )["output"].numpy()
    return embeddings


def make_example(text: str) -> tf.Tensor:
    """Make an example from plain string.

    Args:
        text (str): The input string.

    Returns:
        tf.Tensor: The serialized string example.
    """
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
        "--data_path",
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
    parser.add_argument(
        "--model_location",
        default="local",
        type=str,
        help="The model location: gcs or local",
    )
    args = parser.parse_args()

    if "gcs" in str(args.model_location).lower():
        model_dir = (
            f"gs://motherbrain-pause/model/{args.model}/embed/serving_model_dir/"
        )
    else:
        model_dir = f"./artifacts/model/embed/{args.model}/"
    loaded_model = tf.saved_model.load(model_dir)
    predict_fn = loaded_model.signatures["serving_default"]

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

    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
        "SST2",
        "MR",
        "CR",
        "MPQA",
        "SUBJ",
        "TREC",
        "MRPC",
    ]
    results = se.eval(transfer_tasks)
    print(args.model, results)

    test_result_path = "./artifacts/test"
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    senteval_out_file = "{}/sent_eval_{}.txt".format(test_result_path, args.model)
    with open(senteval_out_file, "w+") as out_file:
        out_file.write(json.dumps(results))
    print("The SentEval test result is exported to {}".format(senteval_out_file))
