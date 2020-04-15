# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import collections
import copy
import logging
import os
import random
from io import open

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from transformers.data.processors import InputExample, InputFeatures
from transformers.file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def oversample(labels, texts):
    max_class_size = max(collections.Counter(labels).values())
    print(f'oversampling. max label count: {max_class_size}')
    dataset = {}
    for t, l in zip(texts, labels):
        if l in dataset:
            dataset[l].append(t)
        else:
            dataset[l] = [t]

    for l, ts in dataset.items():
        if len(ts) < max_class_size:
            q, r = divmod(max_class_size, len(ts))
            ts_original = copy.deepcopy(ts)
            for _ in range(q - 1):
                ts.extend(copy.deepcopy(ts_original))
            ts.extend(random.choices(ts_original, k=r))

    result_texts = []
    result_labels = []
    for l, ts in dataset.items():
        for t in ts:
            result_texts.append(t)
            result_labels.append(l)

    return result_labels, result_texts

import csv

class agnewsProcessor:

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def get_train_examples(self):
    return self._create_examples(os.path.join(self.data_dir, "ag_news_train.csv"))

  def get_dev_examples(self):
    return self._create_examples(os.path.join(self.data_dir, "ag_news_test.csv"))
    # return self._create_examples(os.path.join(data_dir, "test_converted.csv"))


  def get_labels(self):
    """See base class."""
    # return ["0","1"]
    return ['0','1','2','3']

  def _create_examples(self, input_file):
    """Creates examples for the training and dev sets."""
    examples = []
    with open(input_file, encoding='cp949') as f:
      reader = csv.reader(f)
      for i, line in enumerate(reader):

        label =  line[0]
        text_a = line[1]
        if label == '' or text_a == '':
            continue
        examples.append(InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
            # InputExample(guid="unused_id", text_a=text_a, text_b=None, label=label))
    return examples


class OdpProcessor:
    """Processor for the ODP data set"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_file(self, input_file):
        lines = []
        with open(input_file) as f:
            for row in f:
                row = row.rstrip('\n')
                lines.append(row.split(maxsplit=1))
        return lines

    def get_train_examples(self, subset):
        data = self.read_file(os.path.join(self.data_dir, f"tagmynews.train.{subset}"))
        data_t = list(map(list, zip(*data)))
        # data_t = oversample(*data_t)
        data = list(map(list, zip(*data_t)))
        return self._create_examples(data, "train", False)

    def get_dev_examples(self):
        return self._create_examples(
            self.read_file(os.path.join(self.data_dir, "tagmynews.val.all")), "dev", False)

    def get_test_examples(self):
        return self._create_examples(
            self.read_file(os.path.join(self.data_dir, "tagmynews.test.all")), "test", False)

    def get_top5_examples(self):
        return self._create_examples(
            self.read_file(os.path.join(self.data_dir, "nyt.txt")), "top5", False)

    def get_labels(self):
        return ['0','1','2','3','4','5','6']
        # labels = []
        # with open(os.path.join(self.data_dir, 'agnews.label.all')) as f:
        #     for row in f:
        #         row = row.rstrip('\n')
        #         labels.append(row)
        # return labels

    def _create_examples(self, lines, set_type, has_header):
        """Creates examples for the training, dev, and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and has_header:  # header
                continue
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = line[1]


            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def odp_convert_examples_to_features(examples, tokenizer,
                                     max_length=512,
                                     task=None,
                                     label_list=None,
                                     output_mode=None,
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = OdpProcessor()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = odp_output_mode
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield ({'input_ids': ex.input_ids,
                        'attention_mask': ex.attention_mask,
                        'token_type_ids': ex.token_type_ids},
                       ex.label)

        return tf.data.Dataset.from_generator(gen,
                                              ({'input_ids': tf.int32,
                                                'attention_mask': tf.int32,
                                                'token_type_ids': tf.int32},
                                               tf.int64),
                                              ({'input_ids': tf.TensorShape([None]),
                                                'attention_mask': tf.TensorShape([None]),
                                                'token_type_ids': tf.TensorShape([None])},
                                               tf.TensorShape([])))

    return features

def agnews_convert_examples_to_features(examples, tokenizer,
                                     max_length=512,
                                     task=None,
                                     label_list=None,
                                     output_mode=None,
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = agnewsProcessor()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = agnews_output_mode
            logger.info("Using output mode %s for task %s" % (output_mode, task))



    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        if output_mode == "classification":
            label = float(example.label)
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield ({'input_ids': ex.input_ids,
                        'attention_mask': ex.attention_mask,
                        'token_type_ids': ex.token_type_ids},
                       ex.label)

        return tf.data.Dataset.from_generator(gen,
                                              ({'input_ids': tf.int32,
                                                'attention_mask': tf.int32,
                                                'token_type_ids': tf.int32},
                                               tf.int64),
                                              ({'input_ids': tf.TensorShape([None]),
                                                'attention_mask': tf.TensorShape([None]),
                                                'token_type_ids': tf.TensorShape([None])},
                                               tf.TensorShape([])))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def onehot(index_array: np.ndarray, nb_classes: int, dtype=np.float32):
    return np.eye(nb_classes, dtype=dtype)[index_array]


def confusion_per_class(predictions: np.ndarray, labels: np.ndarray, nb_classes: int):
    predictions = onehot(predictions, nb_classes, np.bool8)
    labels = onehot(labels, nb_classes, np.bool8)

    n_predictions = ~predictions
    n_labels = ~labels

    tp_per_class = (predictions & labels).sum(0).astype(np.float32)
    fp_per_class = (predictions & n_labels).sum(0).astype(np.float32)
    fn_per_class = (n_predictions & labels).sum(0).astype(np.float32)
    return tp_per_class, fp_per_class, fn_per_class


def micro_f1_score(tp_per_class, fp_per_class, fn_per_class):
    total_tp = tp_per_class.sum()
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()
    del tp_per_class
    del fp_per_class
    del fn_per_class

    total_precision = total_tp / (total_tp + total_fp + 1e-12)
    total_recall = total_tp / (total_tp + total_fn + 1e-12)

    micro_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-12)
    del total_precision
    del total_recall

    return micro_f1


def www_macro_f1_score(tp_per_class: np.ndarray, fp_per_class: np.ndarray, fn_per_class: np.ndarray):
    is_nonzero_prediction = (tp_per_class + fp_per_class) != 0
    is_nonzero_actual = (tp_per_class + fn_per_class) != 0

    where_nonzero_prediction = np.array(is_nonzero_prediction.nonzero())
    where_nonzero_actual = np.array(is_nonzero_actual.nonzero())
    print(where_nonzero_prediction.shape)
    print(where_nonzero_actual.shape)
    del is_nonzero_prediction, is_nonzero_actual

    precision_per_class = tp_per_class[where_nonzero_prediction] / (
            tp_per_class[where_nonzero_prediction] + fp_per_class[where_nonzero_prediction])
    recall_per_class = tp_per_class[where_nonzero_actual] / (
            tp_per_class[where_nonzero_actual] + fn_per_class[where_nonzero_actual])
    del tp_per_class, fp_per_class, fn_per_class

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    del precision_per_class, recall_per_class

    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    return macro_f1


def odp_compute_metrics(preds, labels, nb_classes):
    assert len(preds) == len(labels)

    tp_per_class, fp_per_class, fn_per_class = confusion_per_class(preds, labels, nb_classes)
    micro_f1 = micro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    macro_f1 = www_macro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


odp_output_mode = "classification"
agnews_output_mode = "classification"

processor = agnewsProcessor
