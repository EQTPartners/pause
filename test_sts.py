"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import os
import argparse
import logging
import pandas as pd
import tensorflow as tf

# The following import is mandatory
import tensorflow_text as text


def run() -> None:
    """Test the trained model (unsupervised) on STS 2012-2016, STSb, and SICK-R datasets."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The trained siamese model",
    )

    opts, _ = parser.parse_known_args()
    print(opts)

    inference_files = [
        "gs://motherbrain-pause/data/inference/data_tfrecord-00000-of-00001.gz",
        "gs://motherbrain-pause/data/inference/data_tfrecord-00001-of-00001.gz",
    ]
    ds_names = ["stsb", "sickr", "sts12", "sts13", "sts14", "sts15", "sts16"]
    feature_description = {
        "uuid": tf.io.FixedLenFeature([], tf.int64),
        "dataset_split": tf.io.FixedLenFeature([], tf.string),
        "sentence": tf.io.FixedLenFeature([], tf.string),
        "match_sentence": tf.io.FixedLenFeature([], tf.string),
        "score": tf.io.FixedLenFeature([], tf.float32),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    print("model={}".format(opts.model))
    model_dir = "./artifacts/model/{}".format(opts.model)
    loaded_model = tf.saved_model.load(model_dir)
    dataset = tf.data.TFRecordDataset(inference_files, compression_type="GZIP")
    f = loaded_model.signatures["serving_default"]

    res = []

    for tfrecord in dataset:
        serialized_example = tfrecord.numpy()
        example = tf.train.Example.FromString(serialized_example)
        uuid = example.features.feature["uuid"].int64_list.value[0]
        score = example.features.feature["score"].float_list.value[0]
        ds = example.features.feature["ds"].bytes_list.value[0].decode()
        info = example.features.feature["info"].bytes_list.value[0].decode()
        pred = f(tf.constant(serialized_example))
        cos_sim = pred["cos_sim"].numpy()[0][0]
        res.append((uuid, score, cos_sim, ds, info))
        if len(res) % 500 == 0:
            print(len(res))

    df = pd.DataFrame(res, columns=["uuid", "score", "cos_sim", "ds", "info"])

    test_result_path = "./artifacts/test"
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    sts_out_file = "{}/sts_{}.txt".format(test_result_path, opts.model)
    with open(sts_out_file, "w+") as out_file:
        for ds in ds_names:
            corr = df[df.ds == ds][["cos_sim", "score"]].corr(method="spearman")
            test_res = "{}={}\n".format(ds, corr.score.iloc[0])
            print(test_res)
            out_file.write(test_res)
    print("The test result is exported to {}".format(sts_out_file))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
