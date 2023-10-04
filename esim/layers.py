"""
Definition of custom layers for the ESIM model.
"""
# Aurelien Coet, 2018.

import torch.nn as nn
import torch
import math

from .utils import sort_by_seq_lens, masked_softmax, weighted_sum  # normal_softmax,


# Class widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/input_variational_dropout.py
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.

    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).

        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.

    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False,
                 total_length=513):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase), "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.total_length = total_length

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.

        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        self._encoder.flatten_parameters()

        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)

        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths.cpu(),
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True,
                                                      total_length=self.total_length)
        # https://pytorch.org/docs/stable/notes/faq.html#my-recurrent-network-doesn-t-work-with-data-parallelism
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class BiAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def __init__(self, dropout=0):
        super(BiAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask,
                ):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
            dropout: dropout rate

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        d = hypothesis_batch.shape[-1]
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous()) / math.sqrt(d)
        similarity_matrix = self.dropout(similarity_matrix)

        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)

        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)

        return attended_premises, attended_hypotheses  # , self_premises, self_hypotheses


class UniAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the scaled soft attention from premises to hypotheses.

    The scaled dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the hypotheses for each element of the premises.
    """

    def __init__(self, dropout=0):
        super(UniAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask,
                ):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
        """
        d = hypothesis_batch.shape[-1]
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous()) / math.sqrt(d)
        similarity_matrix = self.dropout(similarity_matrix)

        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)

        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)

        return attended_premises


class ParamUniAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0):
        super(ParamUniAttention, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.attn = UniAttention(dropout)

    def forward(self,
                hypothesis_batch,
                hypothesis_mask,
                ):
        batch_size = hypothesis_batch.size(0)
        premise_batch = self.weight.expand(batch_size, -1, -1)
        premise_mask = torch.ones_like(hypothesis_mask[:, 0:1])
        return self.attn(premise_batch, premise_mask, hypothesis_batch, hypothesis_mask)


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, hidden_size,
                 num_heads, dropout=0, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, hidden_size, bias=bias)
        self.W_k = nn.Linear(key_size, hidden_size, bias=bias)
        self.W_v = nn.Linear(value_size, hidden_size, bias=bias)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=bias)

    def attention(self, query, query_mask, key, key_mask, value):
        d = key.shape[-1]
        similarity_matrix = query.bmm(key.transpose(2, 1).contiguous()) / math.sqrt(d)
        similarity_matrix = self.dropout(similarity_matrix)

        attn = masked_softmax(similarity_matrix, key_mask)
        attended_query = weighted_sum(value, attn, query_mask)

        return attended_query

    def forward(self, query, query_mask, key, key_mask, value):
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        query_mask = torch.repeat_interleave(query_mask, repeats=self.num_heads, dim=0)
        key_mask = torch.repeat_interleave(key_mask, repeats=self.num_heads, dim=0)

        output = self.attention(query, query_mask, key, key_mask, value)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)

        # sqrt_dim = np.sqrt(premise_batch.size()[2])
        #
        # self_premises_matrix = premise_batch.bmm(premise_batch.transpose(2, 1).contiguous()) / sqrt_dim
        # self_hypotheses_matrix = hypothesis_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous()) / sqrt_dim
        #
        # self_premises_attn = normal_softmax(self_premises_matrix)
        # self_hypotheses_attn = normal_softmax(self_hypotheses_matrix)
        # self_premises = self_premises_attn.bmm(premise_batch)
        # self_hypotheses = self_hypotheses_attn.bmm(hypothesis_batch)

        return attended_premises, attended_hypotheses  # , self_premises, self_hypotheses