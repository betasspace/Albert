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
"""PyTorch ALBERT model. """
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_utils import PreTrainedModel
from .configuration_albert import AlbertConfig
from .file_utils import add_start_docstrings

import pdb

logger = logging.getLogger(__name__)

ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'albert-xlarge-zh': "",
    'albert-large-zh': "",
    'albert-base-zh': "",
}


def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)

    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)

    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name[-13:] == '_embeddings_2':
            pointer = getattr(pointer, 'weight')
            array = np.transpose(array)
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            # print(pointer)
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different:
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as AlbertLayerNorm
except (ImportError, AttributeError) as e:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    AlbertLayerNorm = torch.nn.LayerNorm


class AlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()

        if config.embedding_size == config.hidden_size:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
            self.word_embeddings_2 = None
        else:
            #
            self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
            self.word_embeddings_2 = nn.Linear(config.embedding_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        if self.word_embeddings_2:
            words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # pdb.set_trace()
        # (Pdb) words_embeddings.size(),  position_embeddings.size(),  token_type_embeddings.size()
        # (torch.Size([16, 64, 768]), torch.Size([16, 64, 768]), torch.Size([16, 64, 768]))
        # (Pdb) words_embeddings.min(), words_embeddings.max(), words_embeddings.mean(), words_embeddings.std()
        # (tensor(-0.7094, device='cuda:0'), tensor(0.3465, device='cuda:0'), tensor(-0.0025, device='cuda:0'), tensor(0.0594, device='cuda:0'))
        # (Pdb) position_embeddings.min(), position_embeddings.max(), position_embeddings.mean(), position_embeddings.std()
        # (tensor(-0.1226, device='cuda:0'), tensor(0.1231, device='cuda:0'), tensor(0.0003, device='cuda:0'), tensor(0.0204, device='cuda:0'))
        # (Pdb) token_type_embeddings.min(), token_type_embeddings.max()
        # (tensor(-0.2364, device='cuda:0'), tensor(0.2080, device='cuda:0'))

        return embeddings


class AlbertSelfAttention(nn.Module):
    def __init__(self, config):
        super(AlbertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # pdb.set_trace()
        # (Pdb) self.output_attentions
        # False
        # (Pdb) self.all_head_size
        # 768
        # (Pdb) self.attention_head_size
        # 64


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in AlbertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class AlbertSelfOutput(nn.Module):
    def __init__(self, config):
        super(AlbertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln_type = config.ln_type

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # pdb.set_trace()
        # (Pdb) pp hidden_states.size(), input_tensor.size()
        # (torch.Size([16, 64, 768]), torch.Size([16, 64, 768]))
        if self.ln_type == 'preln':
            hidden_states = hidden_states + input_tensor
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlbertAttention(nn.Module):
    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        self.self = AlbertSelfAttention(config)
        self.output = AlbertSelfOutput(config)
        self.ln_type = config.ln_type

    def forward(self, input_tensor, attention_mask, head_mask=None):
        if self.ln_type == 'preln':
            hidden_state = self.output.LayerNorm(input_tensor)  # pre_ln
            self_outputs = self.self(hidden_state, attention_mask, head_mask)
        else:
            self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class AlbertIntermediate(nn.Module):
    def __init__(self, config):
        super(AlbertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class AlbertOutput(nn.Module):
    def __init__(self, config):
        super(AlbertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln_type = config.ln_type

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.ln_type == 'preln':
            hidden_states = hidden_states + input_tensor
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.ln_type = config.ln_type
        if config.share_type == 'ffn':
            self.attention = nn.ModuleList([AlbertAttention(config) for _ in range(config.num_hidden_layers)])
            self.intermediate = AlbertIntermediate(config)
            self.output = AlbertOutput(config)
        elif config.share_type == 'attention':
            self.attention = AlbertAttention(config)
            self.intermediate = nn.ModuleList([AlbertIntermediate(config) for _ in range(config.num_hidden_layers)])
            self.output = nn.ModuleList([AlbertOutput(config) for _ in range(config.num_hidden_layers)])
        else:
            self.attention = AlbertAttention(config)
            self.intermediate = AlbertIntermediate(config)
            self.output = AlbertOutput(config)

    def forward(self, hidden_states, attention_mask, layer_num, head_mask=None):
        if isinstance(self.attention, nn.ModuleList):
            pdb.set_trace()
            attention_outputs = self.attention[layer_num](hidden_states, attention_mask, head_mask)
        else:
            #--> pdb.set_trace()
            attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        if self.ln_type == 'preln':
            pdb.set_trace()
            attention_output_pre = self.output.LayerNorm(attention_output)
        else:
            # --> pdb.set_trace()
            attention_output_pre = attention_output

        if isinstance(self.intermediate, nn.ModuleList):
            intermediate_output = self.intermediate[layer_num](attention_output_pre)
            pdb.set_trace()
        else:
            # --> pdb.set_trace()
            intermediate_output = self.intermediate(attention_output_pre)

        if isinstance(self.output, nn.ModuleList):
            pdb.set_trace()
            layer_output = self.output[layer_num](intermediate_output, attention_output)
        else:
            #--> pdb.set_trace()
            layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        # pdb.set_trace()
        # (Pdb) layer_output.size()
        # torch.Size([16, 64, 768])
        # (Pdb) type(attention_outputs),len(attention_outputs),attention_outputs[0].size()
        # (<class 'tuple'>, 1, torch.Size([16, 64, 768]))
        # (Pdb) type(outputs), len(outputs), outputs[0].size()
        # (<class 'tuple'>, 1, torch.Size([16, 64, 768]))

        return outputs


class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super(AlbertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_hidden_layers = config.num_hidden_layers
        self.share_type = config.share_type
        # pdb.set_trace()
        # (Pdb) pp type(config.share_type), config.share_type
        # (<class 'str'>, 'all')        
        if config.share_type in ['all', 'ffn', 'attention']:
            self.layer_shared = AlbertLayer(config)
        else:
            self.layer = nn.ModuleList([AlbertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i in range(self.num_hidden_layers):
            if self.share_type in ['all', 'ffn', 'attention']:
                layer_module = self.layer_shared
            else:
                layer_module = self.layer[i]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, i, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class AlbertPooler(nn.Module):
    def __init__(self, config):
        super(AlbertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AlbertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(AlbertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class AlbertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(AlbertLMPredictionHead, self).__init__()
        self.transform = AlbertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.hidden_size != config.embedding_size:
            self.project_layer = nn.Linear(config.hidden_size, config.embedding_size, bias=False)
            self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        else:
            self.project_layer = None
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        if self.project_layer:
            hidden_states = self.project_layer(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class AlbertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(AlbertPreTrainingHeads, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class AlbertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, AlbertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class AlbertModel(AlbertPreTrainedModel):
    r"""
    Examples::
        tokenizer = AlbertTokenizer.from_pretrained('bert-base-uncased')
        model = AlbertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """

    def __init__(self, config):
        super(AlbertModel, self).__init__(config)

        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = AlbertPooler(config)

        self.init_weights()
        # pdb.set_trace();
        # self = AlbertModel(
        #   (embeddings): AlbertEmbeddings(
        #     (word_embeddings): Embedding(21128, 128, padding_idx=0)
        #     (word_embeddings_2): Linear(in_features=128, out_features=768, bias=False)
        #     (position_embeddings): Embedding(512, 768)
        #     (token_type_embeddings): Embedding(2, 768)
        #     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #     (dropout): Dropout(p=0.0, inplace=False)
        #   )
        #   (encoder): AlbertEncoder(
        #     (layer_shared): AlbertLayer(
        #       (attention): AlbertAttention(
        #         (selfatt): AlbertSelfAttention(
        #           (query): Linear(in_features=768, out_features=768, bias=True)
        #           (key): Linear(in_features=768, out_features=768, bias=True)
        #           (value): Linear(in_features=768, out_features=768, bias=True)
        #           (dropout): Dropout(p=0.0, inplace=False)
        #         )
        #         (attout): AlbertSelfOutput(
        #           (dense): Linear(in_features=768, out_features=768, bias=True)
        #           (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #           (dropout): Dropout(p=0.0, inplace=False)
        #         )
        #       )
        #       (intermediate): AlbertIntermediate(
        #         (dense): Linear(in_features=768, out_features=3072, bias=True)
        #       )
        #       (output): AlbertOutput(
        #         (dense): Linear(in_features=3072, out_features=768, bias=True)
        #         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #   )
        #   (pooler): AlbertPooler(
        #     (dense): Linear(in_features=768, out_features=768, bias=True)
        #     (activation): Tanh()
        #   )
        # )


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        # pdb.set_trace()
        # position_ids = None
        # head_mask = None
        # (Pdb) pp input_ids.size(), attention_mask.size(), token_type_ids.size()
        # (torch.Size([16, 64]), torch.Size([16, 64]), torch.Size([16, 64]))

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]

        # pdb.set_trace()
        # (Pdb) pp extended_attention_mask.size()
        # torch.Size([16, 1, 1, 64])

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # pdb.set_trace()
        # (Pdb) head_mask
        # [None, None, None, None, None, None, None, None, None, None, None, None]
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)

        # pdb.set_trace()
        # (Pdb) pp position_ids
        # None

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        # pdb.set_trace()
        # (Pdb) len(encoder_outputs),encoder_outputs[0].size()
        # (1, torch.Size([16, 64, 768]))
        # (Pdb) pooled_output.size()
        # torch.Size([16, 768])
        # ---------------------------------------
        # (Pdb) len(outputs),outputs[0].size(), outputs[1].size()
        # (2, torch.Size([16, 64, 768]), torch.Size([16, 768]))

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings("""Albert Model with two heads on top as done during the pre-training:
    a `masked language modeling` head and a `next sentence prediction (classification)` head. """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class AlbertForPreTraining(AlbertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = AlbertTokenizer.from_pretrained('bert-base-uncased')
        model = AlbertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForPreTraining, self).__init__(config)

        self.bert = AlbertModel(config)
        self.cls = AlbertPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights(config)

        # pdb.set_trace()
        # (Pdb) a
        # self = AlbertForPreTraining(
        #   (bert): AlbertModel(
        #     (embeddings): AlbertEmbeddings(
        #       (word_embeddings): Embedding(21128, 128, padding_idx=0)
        #       (word_embeddings_2): Linear(in_features=128, out_features=768, bias=False)
        #       (position_embeddings): Embedding(512, 768)
        #       (token_type_embeddings): Embedding(2, 768)
        #       (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #       (dropout): Dropout(p=0.0, inplace=False)
        #     )
        #     (encoder): AlbertEncoder(
        #       (layer_shared): AlbertLayer(
        #         (attention): AlbertAttention(
        #           (selfatt): AlbertSelfAttention(
        #             (query): Linear(in_features=768, out_features=768, bias=True)
        #             (key): Linear(in_features=768, out_features=768, bias=True)
        #             (value): Linear(in_features=768, out_features=768, bias=True)
        #             (dropout): Dropout(p=0.0, inplace=False)
        #           )
        #           (attout): AlbertSelfOutput(
        #             (dense): Linear(in_features=768, out_features=768, bias=True)
        #             (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #             (dropout): Dropout(p=0.0, inplace=False)
        #           )
        #         )
        #         (intermediate): AlbertIntermediate(
        #           (dense): Linear(in_features=768, out_features=3072, bias=True)
        #         )
        #         (output): AlbertOutput(
        #           (dense): Linear(in_features=3072, out_features=768, bias=True)
        #           (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #           (dropout): Dropout(p=0.0, inplace=False)
        #         )
        #       )
        #     )
        #     (pooler): AlbertPooler(
        #       (dense): Linear(in_features=768, out_features=768, bias=True)
        #       (activation): Tanh()
        #     )
        #   )
        #   (cls): AlbertPreTrainingHeads(
        #     (predictions): AlbertLMPredictionHead(
        #       (transform): AlbertPredictionHeadTransform(
        #         (dense): Linear(in_features=768, out_features=768, bias=True)
        #         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #       )
        #       (project_layer): Linear(in_features=768, out_features=128, bias=False)
        #       (decoder): Linear(in_features=128, out_features=21128, bias=False)
        #     )
        #     (seq_relationship): Linear(in_features=768, out_features=2, bias=True)
        #   )
        # )

    def tie_weights(self, config):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        if config.embedding_size != config.hidden_size:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
            self._tie_or_clone_data(self.cls.predictions.project_layer,
                                    self.bert.embeddings.word_embeddings_2)
        else:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

    #
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        # add hidden states and attention if they are here
        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


#
@add_start_docstrings("""Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = AlbertTokenizer.from_pretrained('bert-base-uncased')
        model = AlbertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(AlbertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

        # for p in self.bert.parameters():
        #     p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # pdb.set_trace()
        # (Pdb) len(outputs), outputs[0].size(), outputs[1].size()
        # (2, torch.Size([16, 64, 768]), torch.Size([16, 768]))
        # (Pdb) logits.size()
        # torch.Size([16, 2])

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # pdb.set_trace()
        # (Pdb) labels
        # tensor([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], device='cuda:0')

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # pdb.set_trace()
        # (Pdb) loss
        # (Pdb) len(outputs), outputs[0].size(), outputs[0], outputs[1].size()
        # (2, torch.Size([]), tensor(0.7460, device='cuda:0'), torch.Size([16, 2]))
        #         tensor(0.7460, device='cuda:0')

        return outputs  # (loss), logits, (hidden_states), (attentions)
