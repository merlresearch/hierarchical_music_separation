# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, mask, magnitude_spectrogram):
        return mask * magnitude_spectrogram.unsqueeze(-1)


class MaskLayer(nn.Module):
    def __init__(self, input_dim, n_src, n_fft, nonlinearity=torch.sigmoid):
        super().__init__()
        self.mask = EmbeddingLayer(input_dim, n_src, n_fft, nonlinearity, False)

    def __call__(self, x):
        return self.mask(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embed_dim, n_freq_bins, nonlinearity=torch.sigmoid, unit_normalize=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_freq_bins = n_freq_bins

        if nonlinearity is not None and not callable(nonlinearity):
            raise ValueError("mask_nonlinearity must be from chainer.functions!")

        self.nonlinearity = nonlinearity
        self.unit_normalize = unit_normalize

        self.linear = nn.Linear(input_dim, n_freq_bins * embed_dim)

        for name, param in self.linear.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def __call__(self, x):
        batch, _, _ = x.shape
        if self.nonlinearity is not None:
            x = self.nonlinearity(self.linear(x))
        else:
            x = self.linear(x)
        x = x.view(batch, -1, self.embed_dim)
        if self.unit_normalize:
            x = F.normalize(x, dim=-1, p=2)
        return x


class RecurrentStack(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, bidirectional, dropout, rnn_type="lstm"):
        """
        Creates a stack of RNNs used to process an audio sequence represented as (sequence_length, num_features). With
        bidirectional = True, hidden_size = 600, num_layers = 4, rnn_type='lstm', and dropout = .3, this becomes
        the state-of-the-art audio processor used in deep clustering networks, deep attractor networks, etc. Note that
        batch_first is set to True here.

        Args:
            num_features: (int) Number of features being mapped for each frame. Either num_frequencies, or if used with
                MelProjection, num_mels.
            hidden_size: (int) Hidden size of recurrent stack for each layer.
            num_layers: (int) Number of layers in stack.
            bidirectional: (int) True makes this a BiLSTM or a BiGRU. Note that this doubles the hidden size.
            dropout: (float) Dropout between layers.
            rnn_type: (str) LSTM ('lstm') or GRU ('gru').
        """
        super(RecurrentStack, self).__init__()
        if rnn_type not in ["lstm", "gru"]:
            raise ValueError("rnn_type must be one of ['lstm', 'gru']!")

        if rnn_type == "lstm":
            self.add_module(
                "rnn",
                nn.LSTM(
                    num_features,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout,
                ),
            )
        elif rnn_type == "gru":
            self.add_module(
                "rnn",
                nn.GRU(
                    num_features,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout,
                ),
            )

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

    def forward(self, data):
        """
        Args:
            data: Audio representation to be processed. Should be of shape:
                (num_batch, sequence_length, num_features).

        Returns:
            Outputs the features after processing of the RNN. Shape is:
                (num_batch, sequence_length, hidden_size or hidden_size*2 if bidirectional=True)
        """
        shape = data.shape
        data = data.view(shape[0], shape[1], -1)
        self.rnn.flatten_parameters()
        data = self.rnn(data)[0]
        return data


class Embedding(nn.Module):
    def __init__(self, num_features, num_channels, hidden_size, embedding_size, activation):
        """
        Maps output from an audio representation module (e.g. RecurrentStack, DilatedConvolutionalStack) to an embedding
        space. The output shape is (batch_size, sequence_length, num_features, embedding_size). The embeddings can
        be passed through an activation function. If activation is 'softmax' or 'sigmoid', and embedding_size is equal
        to the number of sources, this module can be used to implement a mask inference network (or a mask inference
        head in a Chimera network setup).

        Args:
            num_features: (int) Number of features being mapped for each frame. Either num_frequencies, or if used with
                MelProjection, num_mels if using RecurrentStack. Should be 1 if using DilatedConvolutionalStack.
            hidden_size: (int) Size of output from RecurrentStack (hidden_size) or DilatedConvolutionalStack
                (num_filters). If RecurrentStack is bidirectional, this should be set to 2 * hidden_size.
            embedding_size: (int) Dimensionality of embedding.
            activation: (list of str) Activation functions to be applied. Options are 'sigmoid', 'tanh', 'softmax'.
                Unit normalization can be applied by adding 'unit_norm' in list (e.g. ['sigmoid', unit_norm']).
        """
        super(Embedding, self).__init__()
        self.add_module("linear", nn.Linear(hidden_size, num_features * embedding_size))
        self.num_features = num_features
        self.num_channels = num_channels
        self.activation = activation
        self.embedding_size = embedding_size

        for name, param in self.linear.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, data):
        """
        Args:
            data: output from RecurrentStack or ConvolutionalStack. Shape is:
                (num_batch, sequence_length, hidden_size or num_filters)
        Returns:
            An embedding (with an optional activation) for each point in the representation of shape
            (num_batch, sequence_length, num_features, embedding_size).
        """
        data = self.linear(data)

        # Then this is the output of RecurrentStack and needs to be reshaped a bit.
        # Came in as [num_batch, sequence_length, num_features * embedding_size]
        # Goes out as [num_batch, sequence_length, num_features, embedding_size]
        data = data.view(data.shape[0], data.shape[1], -1, self.num_channels, self.embedding_size)

        if "sigmoid" in self.activation:
            data = torch.sigmoid(data)
        elif "tanh" in self.activation:
            data = torch.tanh(data)
        elif "relu" in self.activation:
            data = torch.relu(data)
        elif "softmax" in self.activation:
            data = torch.softmax(data, dim=-1)

        if "unit_norm" in self.activation:
            data = nn.functional.normalize(data, dim=-1, p=2)

        return data


class BatchNorm(nn.Module):
    def __init__(self, use_batch_norm=True):
        super(BatchNorm, self).__init__()
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.add_module("batch_norm", nn.BatchNorm2d(1))

    def forward(self, data):
        if self.use_batch_norm:
            shape = data.shape
            data = data.view(shape[0], 1, shape[1], -1)
            data = self.batch_norm(data)
            data = data.view(shape)
        return data
