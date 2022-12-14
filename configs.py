# Copyright 2020 The Magenta Authors.
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

# Modification copyright 2022 COMP576-Mice.
# Add Latent VAE model.

# Lint as: python3
"""Configurations for MusicVAE models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.common import merge_hparams
from magenta.models.music_vae import data
from magenta.models.music_vae import data_hierarchical
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE
from base_model import LatentVAE
import magenta.music as mm
from tensorflow.contrib.training import HParams


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name',
     'pretrained_path', 'var_train_pattern', 'encoder_train', 'decoder_train'])):

  def values(self):
    return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
  config_dict = config.values()
  config_dict.update(update_dict)
  return Config(**config_dict)


CONFIG_MAP = {}


# Melody
CONFIG_MAP['cat-mel_2bar_small'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            free_bits=0,
            max_beta=0.2,
            beta_rate=0.99999,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['cat-mel_2bar_big'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
            free_bits=0,
            max_beta=0.5,
            beta_rate=0.99999,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)


CONFIG_MAP['flat-mel_16bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[2048, 2048, 2048],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hierdec-mel_16bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hier-mel_16bar'] = Config(
    model=MusicVAE(
        lstm_models.HierarchicalLstmEncoder(
            lstm_models.BidirectionalLstmEncoder, [16, 16]),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[1024],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['ae-cat-mel_2bar_big'] = Config(
    model=LatentVAE(lstm_models.BidirectionalLstmEncoder(),
                        lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=2,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            encoded_z_size=64,
            latent_encoder_layers=[1024, 256, 64],
            latent_decoder_layers=[64, 256, 1024],
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
            max_beta=0.01,
            beta_rate=0.99999,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
    pretrained_path=None,
    var_train_pattern=['latent'],
    encoder_train=True,
    decoder_train=True
)
