# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from collections import defaultdict

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path,
                      data_path,
                      max_vocabulary_size,
                      tokenizer=None,
                      normalize_digits=True,
                      copy_tokens=[],
                      force=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path) or force:
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_sorted = sorted(vocab, key=vocab.get, reverse=True)
      vocab_list = _START_VOCAB + copy_tokens + vocab_sorted
      vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence,
                          vocabulary,
                          tokenizer=None,
                          normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def sentence_to_copy_ids(source_sentence,
                         target_sentence,
                         vocabulary,
                         tokenizer=None,
                         normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words_source = tokenizer(source_sentence)
    words_target = tokenizer(target_sentence)
  else:
    words_source = basic_tokenizer(source_sentence)
    words_target = basic_tokenizer(target_sentence)
  if normalize_digits:
    # Normalize digits by 0 before looking words up in the vocabulary
    words_source = [_DIGIT_RE.sub(b"0", w) for w in words_source]
    words_target = [_DIGIT_RE.sub(b"0", w) for w in words_target]
  source_word_map = defaultdict(lambda: [])
  for index, word in enumerate(words_source):
    source_word_map[word].append(index)
  result = []
  for w in words_target:
    copy_index_ids = []
    if w in source_word_map:
      copy_indices = ['${}'.format(index) for index in source_word_map[w]]
      copy_index_ids = filter(lambda x: x,
                              [vocabulary.get(copy_index, None) for copy_index in copy_indices])
    # trying to find the token in vocabulary for generating
    if not len(copy_index_ids):
      copy_index_ids.append(vocabulary.get(w, UNK_ID))
    result.append(copy_index_ids)
  return result


def data_to_token_ids(data_path,
                      target_path,
                      vocabulary_path,
                      tokenizer=None,
                      normalize_digits=True,
                      force=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path) or force:
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line),
                                            vocab,
                                            tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def data_to_copy_ids(source_data_path,
                     target_data_path,
                     target_path,
                     vocabulary_path,
                     tokenizer=None,
                     normalize_digits=True,
                     force=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path) or force:
    print("Tokenizing data in (%s, %s)" % (source_data_path, target_data_path))
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(source_data_path, mode="rb") as source_data_file:
      with gfile.GFile(target_data_path, mode="rb") as target_data_file:
        with gfile.GFile(target_path, mode="w") as tokens_file:
          counter = 0
          for source_line, target_line in zip(source_data_file, target_data_file):
            counter += 1
            if counter % 1000 == 0:
              print("  tokenizing line %d" % counter)
            token_ids = sentence_to_copy_ids(tf.compat.as_bytes(source_line),
                                             tf.compat.as_bytes(target_line),
                                             vocab,
                                             tokenizer,
                                             normalize_digits)
            tokens_file.write(" ".join([';'.join(map(str, tokens))
                                        for tokens in token_ids]) + "\n")


def create_copy_vocabulary(in_copy_vocabulary_size):
    return ['${}'.format(index)
            for index in xrange(in_copy_vocabulary_size)]


def prepare_data(data_dir,
                 from_train_path,
                 to_train_path,
                 from_dev_path,
                 to_dev_path,
                 from_vocabulary_size,
                 to_vocabulary_size,
                 tokenizer=None,
                 copy_tokens_number=0,
                 force=False):
  """Preapre all necessary files that are required for the training.

    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.
  copy_vocabulary = create_copy_vocabulary(copy_tokens_number)
  # forcing model to copy only
  to_vocabulary_size = len(copy_vocabulary) + len(_START_VOCAB)
  to_vocab_path = os.path.join(data_dir, "vocab.to")
  from_vocab_path = os.path.join(data_dir, "vocab.from")

  create_vocabulary(to_vocab_path,
                    to_train_path,
                    to_vocabulary_size,
                    tokenizer,
                    copy_tokens=copy_vocabulary,
                    force=force)
  create_vocabulary(from_vocab_path,
                    from_train_path,
                    from_vocabulary_size,
                    tokenizer,
                    force=force)

  from_train_ids_path, to_train_ids_path, to_train_target_ids_path = make_dataset(from_train_path,
                                                                                  to_train_path,
                                                                                  from_vocab_path,
                                                                                  to_vocab_path,
                                                                                  tokenizer=None,
                                                                                  force=False)
  from_dev_ids_path, to_dev_ids_path, to_dev_target_ids_path = make_dataset(from_dev_path,
                                                                            to_dev_path,
                                                                            from_vocab_path,
                                                                            to_vocab_path,
                                                                            tokenizer=None,
                                                                            force=False)
  return (from_train_ids_path,
          to_train_ids_path,
          to_train_target_ids_path,
          from_dev_ids_path,
          to_dev_ids_path,
          to_dev_target_ids_path,
          from_vocab_path,
          to_vocab_path)


def make_dataset(from_path, to_path, from_vocab_path, to_vocab_path, tokenizer=None, force=False):
  # Create token ids
  # encoder inputs - just ids from the encoder vocabulary
  from_ids_path = from_path + ".ids.from"
  data_to_token_ids(from_path,
                    from_ids_path,
                    from_vocab_path,
                    tokenizer,
                    force=force)
  # decoder inputs - decoder sequence ids from encoder vocabulary
  # (for feeding into the decoder at each time step)
  to_ids_path = to_path + ".ids.from"
  data_to_token_ids(to_path,
                    to_ids_path,
                    from_vocab_path,
                    tokenizer,
                    force=force)
  # decoder targets - decoder sequences as indices in the encoder sequences
  to_target_ids_path = to_path + ".ids.to"
  data_to_copy_ids(from_path,
                   to_path,
                   to_target_ids_path,
                   to_vocab_path,
                   tokenizer,
                   force=force)
  return from_ids_path, to_ids_path, to_target_ids_path 

