from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based import neuron, layer,surrogate
import numpy as np
from spikingjelly.activation_based.neuron import LIFNode,ParametricLIFNode
from module.conv import Transpose,PointwiseConv1d,DepthwiseConv1d
from spikingjelly.datasets import padded_sequence_mask
from module.spikcommander_backbone import Backbone

class SEE(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2p, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True)
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias)

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False,
                                   step_mode='m')

        if self.config.use_bn:
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        self.lif1 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.lif2 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )


    def forward(self, x):
        # batch, time, dim =>  batch, dim, time
        x = self.trans1(x)
        x = self.pwconv(x)
        x = self.dwconv(x)
        # batch, dim, time =>  time, batch, dim
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn1(x)
        x = self.lif1(x)
        if self.config.use_dp:
            x = self.dropout1(x)
        x_res = x
        x = self.linear(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)
        x = x + x_res

        return x



class SpikCommander(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config


        self.see = SEE(config, kernel_size=7)


        self.blocks = nn.ModuleList(
            [
                Backbone(
                    config=self.config,
                    dim=self.config.n_hidden_neurons,
                    num_heads=self.config.num_heads,
                    init_tau = self.config.init_tau,
                    spike_mode=self.config.spike_mode,
                    layers=j,
                )
                for j in range(self.config.depths)
            ]
        )


        if self.config.use_dp:
            self.final_dp = layer.Dropout(self.config.dropout_l, step_mode='m')

        self.head = layer.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=False, step_mode='m')


        self._reset_parameters()


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x, attention_mask):

        x = self.see(x)

        for module in self.blocks:
            x = module(x, attention_mask)

        if self.config.use_dp:
            x = self.head(self.final_dp(x))
        else:
            x = self.head(x)


        return x


