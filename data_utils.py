"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Feature and label specification (dictionary) of the pre-processed dataset
feature_spec = {
    "gold_label": tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=None),
    "label": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=None),
    "match_sentence": tf.io.FixedLenFeature(
        shape=[1], dtype=tf.string, default_value=None
    ),
    "sentence": tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=None),
    "uuid": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=None),
}


def train_files(prec: str = "50") -> list:
    """Obtain a list of pre-processed dataset file patterns for training.

    Args:
        prec (str, optional): The percentage of labels to use. Defaults to "50".

    Returns:
        list: The TFRecord file patterns (entailment, contradiction, neutral, unlabeled) for training.
    """
    return [
        "gs://motherbrain-pause/data/{}p/entailment/*".format(prec),
        "gs://motherbrain-pause/data/{}p/contradiction/*".format(prec),
        "gs://motherbrain-pause/data/{}p/neutral/*".format(prec),
        "gs://motherbrain-pause/data/{}p/unl/*".format(prec),
    ]


def eval_files(prec: str = "50") -> list:
    """Obtain a list of pre-processed dataset file patterns for evaluation.

    Args:
        prec (str, optional): The percentage here should match with train_files.

    Returns:
        list: The TFRecord file patterns for evaluation during training.
    """
    return ["gs://motherbrain-pause/data/{}p/eval/*".format(prec)]


def get_file(file_pattern: list, sub_type: str = None) -> list:
    """Get a subset from file patterns that belong to a sub-type.
    If no sub-type is specified, return all file patterns.

    Args:
        file_pattern (list): The input file patterns
        sub_type (str, optional): A string to search in file patterns. Defaults to None.

    Raises:
        ValueError: No file pattern matches the sub-type provided.

    Returns:
        list: A filtered sub list of file patterns.
    """
    if sub_type is None:
        return file_pattern
    result = []
    for entry in file_pattern:
        if sub_type in entry:
            result.append(entry)
    if len(result) < 1:
        raise ValueError(
            "No file found for sub-type {}: {}".format(sub_type, file_pattern)
        )
    else:
        return result


def make_dataset(
    feature_spec: dict,
    file_pattern: list,
    batch_size: int,
    label_key: str,
    prior: float = None,
    training: bool = True,
) -> tf.data.Dataset:
    """Construct a dataset for training or evaluation

    Args:
        feature_spec (dict): The feature specification of input TFRecord files.
        file_pattern (list): The input file patterns.
        batch_size (int): Batch size.
        label_key (str): The key of the label.
        prior (float, optional): The prior hyper-parameter. Defaults to None (should not be None for training).
        training (bool, optional): Indicate if this is for training. Defaults to True.

    Returns:
        tf.data.Dataset: The constructed dataset.
    """

    def _parse_function(example_proto):
        transformed_features = tf.io.parse_single_example(example_proto, feature_spec)
        transformed_features.pop(label_key)
        transformed_label = transformed_features.pop("label")
        return transformed_features, transformed_label

    def _get_ds(file_pattern, sub_type):
        _ds = tf.data.TFRecordDataset(
            filenames=tf.data.Dataset.list_files(
                get_file(file_pattern, sub_type=sub_type)
            ),
            compression_type="GZIP",
        )
        _ds = _ds.shuffle(200000)
        return _ds.repeat()

    if training:
        entailment_ds = _get_ds(file_pattern, sub_type="entailment")
        contradiction_ds = _get_ds(file_pattern, sub_type="contradiction")
        neutral_ds = _get_ds(file_pattern, sub_type="neutral")
        unl_ds = _get_ds(file_pattern, sub_type="unl")

        _pr = prior / 3.0
        dataset = tf.data.experimental.sample_from_datasets(
            [entailment_ds, contradiction_ds, neutral_ds, unl_ds],
            weights=[_pr, _pr, _pr, 1 - prior],
        )
    else:
        dataset = tf.data.TFRecordDataset(
            filenames=tf.data.Dataset.list_files(
                get_file(file_pattern, sub_type="eval")
            ),
            compression_type="GZIP",
        )

    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
