from functools import partial
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based import neuron, layer,surrogate
import numpy as np
from spikingjelly.activation_based.neuron import LIFNode,ParametricLIFNode
from module.conv import Transpose,PointwiseConv1d,DepthwiseConv1d
from spikingjelly.datasets import padded_sequence_mask
# from module.conformer_9312_rp_gate_qk_first_attention_mask_pw_dw_pwg_qkformer_ssa import MS_Block_Conv
# from module.conformer_9312_rp_gate_qk_first_attention_mask_pw_dw_pwg_qkformer_ssa_refine_ffn import MS_Block_Conv
# from module.conformer_9312_rp_gate_qk_first_attention_mask_pw_dw_pwg_qkformer_ssa_dw import MS_Block_Conv
# from module.conformer_9312_rp_gate_qk_first_attention_mask_pw_dw_pwg_qkformer_ssa_pw_dw import MS_Block_Conv
# from module.conformer_9312_rp_gate_qk_first_attention_mask_pw_dw_pwg_qkformer_ssa_refine_ffn_attn_conv import MS_Block_Conv
# from module.conformer_9312_rp_gate_qk_first_attention_mask_pw_dw_pwg_qkformer_ssa_refine_ffn_attn_conv_token_shift import MS_Block_Conv
# from module.qka_ssa_refine_ffn_attn_conv import MS_Block_Conv
from module.qka_ssa_refine_ffn_attn_conv_longformer import MS_Block_Conv # STASA

from spikingjelly.activation_based import rnn
from spikingjelly.activation_based.rnn import SpikingGRUCell, SpikingGRU, SpikingRNNCellBase,SpikingRNNBase


class SpikingGRUCellWithBN(SpikingRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias=True, dropout_p=0.0,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        super().__init__(input_size, hidden_size, bias)

        # --- 1. 定义核心层 ---
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        # --- 2. 实例化 BN 和 Dropout 层 ---
        # 对 input-to-hidden 的变换结果进行 BN
        # self.bn_ih = nn.BatchNorm1d(3 * hidden_size)
        # 对 hidden-to-hidden 的变换结果进行 BN
        # self.bn_hh = nn.BatchNorm1d(3 * hidden_size)
        # 实例化 Dropout 层
        self.dropout = nn.Dropout(dropout_p)

        # --- 3. 设置代理梯度函数 ---
        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.reset_parameters()

    def forward(self, x: torch.Tensor, h=None):
        if h is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)

        # --- 4. 在 forward 中应用 BN ---
        # 先进行线性变换，然后应用BN
        # y_ih_bn = self.bn_ih(self.linear_ih(x))
        y_ih_bn = self.linear_ih(x)
        # y_ih_bn = self.dropout(y_ih_bn)

        # y_hh_bn = self.bn_hh(self.linear_hh(h))
        y_hh_bn = self.linear_hh(h)
        # y_hh_bn = self.dropout(y_hh_bn)

        # 将结果切分为 r, z, n 对应的部分
        y_ih = torch.split(y_ih_bn, self.hidden_size, dim=1)
        y_hh = torch.split(y_hh_bn, self.hidden_size, dim=1)

        # 计算门控和候选状态
        r = self.surrogate_function1(y_ih[0] + y_hh[0])
        z = self.surrogate_function1(y_ih[1] + y_hh[1])

        if self.surrogate_function2 is None:
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
        else:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking
            n = self.surrogate_function2(y_ih[2] + r * y_hh[2])

        # 计算新的隐藏状态
        h_new = (1. - z) * n + z * h

        # --- 5. 在 forward 中应用 Dropout ---
        h_new = self.dropout(h_new)

        return h_new


class CustomSpikingGRU(SpikingRNNBase):
    def __init__(self, *args, cell_dropout_p=0.0, **kwargs):
        self.cell_dropout_p = cell_dropout_p
        super().__init__(*args, **kwargs)

    def create_cells(self, *args, **kwargs):
        # 从kwargs里把原来的 surrogate_function 取出来
        surrogate_function1 = kwargs.get('surrogate_function1', surrogate.Erf(alpha=5.0))
        surrogate_function2 = kwargs.get('surrogate_function2', None)

        # 正常造
        if self.bidirectional:
            cells = []
            cells_reverse = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias,
                                          dropout_p=self.cell_dropout_p,
                                          surrogate_function1=surrogate_function1,
                                          surrogate_function2=surrogate_function2))
            cells_reverse.append(self.base_cell()(self.input_size, self.hidden_size, self.bias,
                                                  dropout_p=self.cell_dropout_p,
                                                  surrogate_function1=surrogate_function1,
                                                  surrogate_function2=surrogate_function2))
            for i in range(self.num_layers - 1):
                in_size = self.hidden_size * 2
                cells.append(self.base_cell()(in_size, self.hidden_size, self.bias,
                                              dropout_p=self.cell_dropout_p,
                                              surrogate_function1=surrogate_function1,
                                              surrogate_function2=surrogate_function2))
                cells_reverse.append(self.base_cell()(in_size, self.hidden_size, self.bias,
                                                      dropout_p=self.cell_dropout_p,
                                                      surrogate_function1=surrogate_function1,
                                                      surrogate_function2=surrogate_function2))
            return nn.Sequential(*cells), nn.Sequential(*cells_reverse)
        else:
            cells = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias,
                                          dropout_p=self.cell_dropout_p,
                                          surrogate_function1=surrogate_function1,
                                          surrogate_function2=surrogate_function2))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size, self.hidden_size, self.bias,
                                              dropout_p=self.cell_dropout_p,
                                              surrogate_function1=surrogate_function1,
                                              surrogate_function2=surrogate_function2))
            return nn.Sequential(*cells)

    @staticmethod
    def base_cell():
        return SpikingGRUCellWithBN

    @staticmethod
    def states_num():
        return 1


class SpikingEmbed(nn.Module):
    def __init__(self, config):
        super(SpikingEmbed, self).__init__()
        self.config = config

        self.conv = nn.Conv1d(config.n_inputs, config.n_hidden_neurons, kernel_size=3, stride=1, bias=config.bias, padding=1)
        if self.config.use_bn:
            self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0,2,1)
        self.trans2 = Transpose(2,0,1)

        if self.config.spike_mode == 'lif':
            self.lif = LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )
        elif self.config.spike_mode == 'plif':
            self.lif = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

    def forward(self, x):
        x = self.trans1(x)
        x = self.conv(x)
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn(x)
        x = self.lif(x)
        if self.config.use_dp:
            x = self.dropout(x)
        return x

class SpikingEmbedv2(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True)
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                                  bias=config.use_dw_bias)

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias

        if self.config.use_bn:
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
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

        x = self.linear(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)

        return x


def channel_shuffle(x, groups):
    # x: [T, B, C]
    T, B, C = x.shape
    assert C % groups == 0, "Number of channels must be divisible by groups"
    x = x.reshape(T, B, groups, C // groups)       # [T, B, G, C//G]
    x = x.permute(0, 1, 3, 2)                      # [T, B, C//G, G]
    x = x.reshape(T, B, C)                         # [T, B, C]
    return x

class SpikingEmbedv2p(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2p, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, # False，True
                                   step_mode='m')  # self.config.bias

        if self.config.use_bn:
            # self.bn0 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
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
        # # batch, dim, time =>  time, batch, dim
        # x = self.trans2(x)
        # if self.config.use_bn:
        #     x = self.bn0(x)
        # x = self.trans3(x)  # time, batch, dim => batch, dim, time
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


class SpikingEmbedv2pgated(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2pgated, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        self.linear = layer.Linear(self.config.n_hidden_neurons//2 , self.config.n_hidden_neurons//2, bias=False, # False，True
                                   step_mode='m')  # self.config.bias

        if self.config.use_bn:
            # self.bn0 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons//2, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
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
        # # batch, dim, time =>  time, batch, dim
        # x = self.trans2(x)
        # if self.config.use_bn:
        #     x = self.bn0(x)
        # x = self.trans3(x)  # time, batch, dim => batch, dim, time
        x = self.dwconv(x)
        # batch, dim, time =>  time, batch, dim
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn1(x)
        x = self.lif1(x)
        if self.config.use_dp:
            x = self.dropout1(x)
        # split x
        x_res, x_gate = torch.split(x, self.config.n_hidden_neurons // 2, dim=-1)  # [T, B, C//2], [T, B, C//2]
        # x_gate, x_res  = torch.split(x, self.config.n_hidden_neurons // 2, dim=-1)  # [T, B, C//2], [T, B, C//2]
        x_gate = self.linear(x_gate)
        x_gate = self.bn2(x_gate)
        x_gate = self.lif2(x_gate)
        x_gate = self.dropout2(x_gate)
        # concat
        x = torch.cat([x_res, x_gate], dim=-1)
        # x = torch.cat([x_gate, x_res], dim=-1)
        return x

class SpikingEmbedv2pgru(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2pgru, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        # self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, # False，True
        #                            step_mode='m')  # self.config.bias

        if self.config.use_bn:
            # self.bn0 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

        self.sgru = rnn.SpikingGRU(  # SpikingGRUv2
                    # config,
                    self.config.n_hidden_neurons,
                    self.config.n_hidden_neurons,
                    num_layers=1,
                    bias=False,
                    dropout_p=config.dropout_l,
                    invariant_dropout_mask=False,
                    bidirectional=False,
                    surrogate_function1=config.surrogate_function,  # surrogate.Erf(), config.surrogate_function
                    surrogate_function2=None
                )

        # self.sgru = CustomSpikingGRU(
        #     input_size=self.config.n_hidden_neurons,
        #     hidden_size=self.config.n_hidden_neurons,
        #     num_layers=1,  # 改为2层来演示层间 dropout 的作用
        #     bias=False,
        #
        #     # 这个 dropout_p 用于 num_layers > 1 时，在两层 GRU 的输出之间施加 dropout
        #     # 由 SpikingRNNBase 的 forward 方法处理
        #     dropout_p=config.dropout_l,
        #
        #     # 这是我们新增的关键字，用于控制 Cell 内部的 BN 和 Dropout
        #     cell_dropout_p=config.dropout_l,
        #
        #     bidirectional=False,
        #     surrogate_function1=surrogate.Erf(),
        #     surrogate_function2=None
        # )

    def forward(self, x):
        # batch, time, dim =>  batch, dim, time
        x = self.trans1(x)
        x = self.pwconv(x)
        # # batch, dim, time =>  time, batch, dim
        # x = self.trans2(x)
        # if self.config.use_bn:
        #     x = self.bn0(x)
        # x = self.trans3(x)  # time, batch, dim => batch, dim, time
        x = self.dwconv(x)
        # batch, dim, time =>  time, batch, dim
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn1(x)
        x = self.lif1(x)
        if self.config.use_dp:
            x = self.dropout1(x)
        x_res = x
        # x = self.linear(x)
        # x = self.bn2(x)
        # x = self.lif2(x)
        x, state = self.sgru(x)
        # x = self.dropout2(x)

        x = x + x_res

        return x

class SpikingEmbedv2ptsm(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2ptsm, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, # False，True
                                   step_mode='m')  # self.config.bias

        if self.config.use_bn:
            # self.bn0 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )
        if self.config.dataset =='shd':
            self.tsm = TemporalShift1Dv2(config,fold_div=32)
        else:
            self.tsm = TemporalShift1Dv2(config, fold_div=32)

    def forward(self, x):
        # x_res = x
        x =self.tsm(x)
        # x = x * x_res


        # batch, time, dim =>  batch, dim, time
        x = self.trans1(x)
        x = self.pwconv(x)
        # # batch, dim, time =>  time, batch, dim
        # x = self.trans2(x)
        # if self.config.use_bn:
        #     x = self.bn0(x)
        # x = self.trans3(x)  # time, batch, dim => batch, dim, time
        x = self.dwconv(x)
        # batch, dim, time =>  time, batch, dim
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn1(x)

        x =self.tsm(x)

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

class SpikingEmbedv2pcs(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2pcs, self).__init__()

        # 在linear残差中加入channel shuffle

        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, # False，True
                                   step_mode='m')  # self.config.bias

        if self.config.use_bn:
            # self.bn0 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
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
        # # batch, dim, time =>  time, batch, dim
        # x = self.trans2(x)
        # if self.config.use_bn:
        #     x = self.bn0(x)
        # x = self.trans3(x)  # time, batch, dim => batch, dim, time
        x = self.dwconv(x)
        # batch, dim, time =>  time, batch, dim
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn1(x)
        x = self.lif1(x)
        if self.config.use_dp:
            x = self.dropout1(x)
        x_res = x
        x_res = channel_shuffle(x_res, groups=self.config.n_hidden_neurons // 16)
        x = self.linear(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)
        x = x + x_res

        return x

class SpikingEmbedv2pd(nn.Module):
    def __init__(self, config, kernel_size=5):
        super(SpikingEmbedv2pd, self).__init__()

        # 在DW中加入膨胀系数

        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1, dilation=2,
                                      padding=4,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, # False，True
                                   step_mode='m')  # self.config.bias

        if self.config.use_bn:
            # self.bn0 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
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
        # # batch, dim, time =>  time, batch, dim
        # x = self.trans2(x)
        # if self.config.use_bn:
        #     x = self.bn0(x)
        # x = self.trans3(x)  # time, batch, dim => batch, dim, time
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

class SpikingEmbedv2pp(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2pp, self).__init__()
        # 相较于SpikingEmbedv2p，在最开始添加了Linear

        self.config = config

        self.pwconv = PointwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, # False，True
                                   step_mode='m')  # self.config.bias

        self.linear0 = layer.Linear(self.config.n_inputs, self.config.n_hidden_neurons, bias=False, # False，True
                                   step_mode='m')  # self.config.bias

        if self.config.use_bn:
            self.bn0 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout0 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)
        self.trans4 = Transpose(1, 0, 2)

        if self.config.spike_mode == 'lif':
            self.lif0 = LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

    def forward(self, x):

        # x_res = x
        # batch, time, dim =>  time, batch, dim (1,0,2)
        x = self.trans4(x)
        x = self.linear0(x)
        x = self.bn0(x)
        x = self.lif0(x)
        x = self.dropout0(x)
        # x = x + x_res

        # batch, time, dim =>  batch, dim, time
        x = self.trans3(x)
        x = self.pwconv(x)
        # # batch, dim, time =>  time, batch, dim
        # x = self.trans2(x)
        # if self.config.use_bn:
        #     x = self.bn0(x)
        # x = self.trans3(x)  # time, batch, dim => batch, dim, time
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

class SpatioTemporalInteraction(nn.Module):
    def __init__(self, config, reduction='mean'):
        """
        Parameters:
        - mode: 'residual' or 'linear'
        - reduction: 'mean' or 'sum' for aggregation
        """
        super().__init__()
        self.reduction = reduction
        self.config = config

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

        # if self.reduction == 'mean':

        self.spaital_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.temporal_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias
        if self.config.use_bn:
            self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

    def forward(self, x):
        """
        x: [T, B, N]
        """
        if self.reduction == 'mean':
            spatial_sum = x.mean(dim=-1, keepdim=True)  # [T, B, 1]
            spatial_sum = self.spaital_lif(spatial_sum)

            temporal_sum = x.mean(dim=0, keepdim=True)  # [1, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)

        else:
            spatial_sum = x.sum(dim=-1, keepdim=True)   # [T, B, 1]
            temporal_sum = x.sum(dim=0, keepdim=True)   # [1, B, N]

            # Compute modulated features
            spatial_sum = x * spatial_sum        # [T, B, N]
            # spatial_sum = self.spaital_lif(spatial_sum)
            temporal_sum = x * temporal_sum      # [T, B, N]
            # temporal_sum = self.temporal_lif(temporal_sum)
            sum = spatial_sum * temporal_sum

        x = x + sum
        # x = x + spatial_sum + temporal_sum
        x = self.lif1(x)

        x = self.linear(x)
        x = self.bn(x)
        x = self.lif2(x)
        x = self.dropout(x)
        return x


class SpatioTemporalInteractionv2(nn.Module):
    def __init__(self, config, reduction='mean'):
        """
        Parameters:
        - mode: 'residual' or 'linear'
        - reduction: 'mean' or 'sum' for aggregation
        """
        super().__init__()
        self.reduction = reduction
        self.config = config

        # self.lif1 = LIFNode(
        #     tau=config.init_tau,
        #     v_threshold=config.v_threshold, v_reset=config.v_reset,
        #     surrogate_function=config.surrogate_function,
        #     detach_reset=config.detach_reset,
        #     step_mode='m',
        #     decay_input=False,
        #     store_v_seq=False,
        #     backend=config.backend
        # )

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

        # if self.reduction == 'mean':

        self.spaital_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.temporal_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias

        if self.config.use_bn:
            self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

    def forward(self, x):
        """
        x: [T, B, N]
        """
        x_res = x

        if self.reduction == 'mean':
            spatial_sum = x.mean(dim=-1, keepdim=True)  # [T, B, 1]
            spatial_sum = self.spaital_lif(spatial_sum)

            temporal_sum = x.mean(dim=0, keepdim=True)  # [1, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)

        else:
            spatial_sum = x.sum(dim=-1, keepdim=True)   # [T, B, 1]
            temporal_sum = x.sum(dim=0, keepdim=True)   # [1, B, N]

            x = self.linear(x)
            x = self.bn(x)
            x = self.lif2(x)
            x = self.dropout(x)

            # Compute modulated features
            spatial_sum = x * spatial_sum        # [T, B, N]
            spatial_sum = self.spaital_lif(spatial_sum)
            temporal_sum = x * temporal_sum      # [T, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)
            sum = spatial_sum * temporal_sum

        x = x_res + sum
        # x = x + spatial_sum + temporal_sum
        # x = self.lif1(x)


        return x

class SpatioTemporalInteractionv3(nn.Module):
    def __init__(self, config, reduction='mean'):
        """
        Parameters:
        - mode: 'residual' or 'linear'
        - reduction: 'mean' or 'sum' for aggregation
        """
        super().__init__()
        self.reduction = reduction
        self.config = config

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

        # if self.reduction == 'mean':

        self.spaital_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.temporal_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias

        if self.config.use_bn:
            self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

    def forward(self, x):
        """
        x: [T, B, N]
        """
        if self.reduction == 'mean':
            spatial_sum = x.mean(dim=-1, keepdim=True)  # [T, B, 1]
            spatial_sum = self.spaital_lif(spatial_sum)

            temporal_sum = x.mean(dim=0, keepdim=True)  # [1, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)

        else:
            spatial_sum = x.sum(dim=-1, keepdim=True)   # [T, B, 1]
            temporal_sum = x.sum(dim=0, keepdim=True)   # [1, B, N]

            # Compute modulated features
            spatial_sum = x * spatial_sum        # [T, B, N]
            spatial_sum = self.spaital_lif(spatial_sum)
            temporal_sum = x * temporal_sum      # [T, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)
            sum = spatial_sum * temporal_sum

        sum = self.linear(sum)
        sum = self.bn(sum)
        sum = self.lif2(sum)
        sum = self.dropout(sum)

        x = x + sum

        return x


class SpatioTemporalInteractionv4(nn.Module):
    def __init__(self, config, reduction='mean'):
        """
        Parameters:
        - mode: 'residual' or 'linear'
        - reduction: 'mean' or 'sum' for aggregation
        """
        super().__init__()
        self.reduction = reduction
        self.config = config


        # if self.reduction == 'mean':

        self.spaital_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.temporal_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.linear1 = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias
        if self.config.use_bn:
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

        self.lif1 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend)


        self.linear2 = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias
        if self.config.use_bn:
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

        self.lif2 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend)


    def forward(self, x):
        """
        x: [T, B, N]
        """
        if self.reduction == 'mean':
            spatial_sum = x.mean(dim=-1, keepdim=True)  # [T, B, 1]
            spatial_sum = self.spaital_lif(spatial_sum)

            temporal_sum = x.mean(dim=0, keepdim=True)  # [1, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)

        else:
            spa_x = self.linear1(x)
            spa_x = self.bn1(spa_x)
            spa_x = self.lif1(spa_x)
            spatial_sum = spa_x.sum(dim=-1, keepdim=True)   # [T, B, 1]

            tem_x = self.linear2(x)
            tem_x = self.bn2(tem_x)
            tem_x = self.lif2(tem_x)
            temporal_sum = tem_x.sum(dim=0, keepdim=True)   # [1, B, N]

            # Compute modulated features
            spatial_sum = x * spatial_sum        # [T, B, N]
            spatial_sum = self.spaital_lif(spatial_sum)
            temporal_sum = x * temporal_sum      # [T, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)
            sum = spatial_sum * temporal_sum

        x = x + sum

        return x



class SpatioTemporalInteractionv5(nn.Module):
    def __init__(self, config, reduction='mean'):
        """
        Parameters:
        - mode: 'residual' or 'linear'
        - reduction: 'mean' or 'sum' for aggregation
        """
        super().__init__()
        self.reduction = reduction
        self.config = config

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

        # if self.reduction == 'mean':

        self.spaital_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.temporal_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold, v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias
        if self.config.use_bn:
            self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

    def forward(self, x):
        """
        x: [T, B, N]
        """
        x_res = x
        x = self.linear(x)
        x = self.bn(x)
        x = self.lif2(x)
        x = self.dropout(x)

        if self.reduction == 'mean':
            spatial_sum = x.mean(dim=-1, keepdim=True)  # [T, B, 1]
            spatial_sum = self.spaital_lif(spatial_sum)

            temporal_sum = x.mean(dim=0, keepdim=True)  # [1, B, N]
            temporal_sum = self.temporal_lif(temporal_sum)

        else:
            spatial_sum = x.sum(dim=-1, keepdim=True)   # [T, B, 1]
            temporal_sum = x.sum(dim=0, keepdim=True)   # [1, B, N]


            # Compute modulated features
            spatial_sum = x * spatial_sum        # [T, B, N]
            # spatial_sum = self.spaital_lif(spatial_sum)

            temporal_sum = x * temporal_sum      # [T, B, N]
            # temporal_sum = self.temporal_lif(temporal_sum)

            # sum = spatial_sum * temporal_sum

        # x = x + spatial_sum + temporal_sum
        sum =  spatial_sum + temporal_sum
        sum = self.lif1(sum)

        x  = x_res + sum
        return x

class SpikingEmbedv3(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv3, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True)
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                                  bias=config.use_dw_bias)


        if self.config.use_bn:
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)

        if self.config.spike_mode == 'lif':
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


        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

        # self.stinteraction = SpatioTemporalInteraction(config, reduction='sum')
        # self.stinteraction = SpatioTemporalInteraction(config, reduction='mean')
        # self.stinteraction = SpatioTemporalInteractionv2(config, reduction='sum')
        # self.stinteraction = SpatioTemporalInteractionv3(config, reduction='sum')
        # self.stinteraction = SpatioTemporalInteractionv4(config, reduction='sum')
        self.stinteraction = SpatioTemporalInteractionv5(config, reduction='sum')

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

        x = self.stinteraction(x)

        return x


class TemporalShift1D(nn.Module):
    def __init__(self, fold_div=4):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x):
        # x: [T, B, C]
        T, B, C = x.size()
        fold = C // self.fold_div
        out = torch.zeros_like(x)

        # 后移：第 1 段
        out[1:, :, :fold] = x[:-1, :, :fold]

        # 不动：中间部分
        out[:, :, fold:(self.fold_div - 1) * fold] = x[:, :, fold:(self.fold_div - 1) * fold]

        # 前移：最后一段
        out[:-1, :, (self.fold_div - 1) * fold:] = x[1:, :, (self.fold_div - 1) * fold:]

        return out


class TemporalShift1Dv2(nn.Module):
    def __init__(self, config, fold_div=4):
        super().__init__()
        self.fold_div = fold_div

        # self.lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold, v_reset=config.v_reset,
        #                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
        #                        step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)
    def forward(self, x):
        # x: [T, B, C]
        T, B, C = x.size()
        fold = C // self.fold_div
        out = torch.zeros_like(x)

        # 后移：第 1 段
        # out[1:, :, :fold] = x[:-1, :, :fold] + x[1:, :, :fold]
        out[1:, :, :fold] = x[:-1, :, :fold]

        # 不动：中间部分
        out[:, :, fold:(self.fold_div - 1) * fold] = x[:, :, fold:(self.fold_div - 1) * fold]

        # 前移：最后一段
        # out[:-1, :, (self.fold_div - 1) * fold:] = x[1:, :, (self.fold_div - 1) * fold:] + x[:-1, :, (self.fold_div - 1) * fold:]
        out[:-1, :, (self.fold_div - 1) * fold:] = x[1:, :, (self.fold_div - 1) * fold:]

        # out = self.lif(out)

        return out


class TemporalShift1Dv3(nn.Module):
    def __init__(self, config, fold_div=4):
        super().__init__()
        self.fold_div = fold_div

        # self.lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold, v_reset=config.v_reset,
        #                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
        #                        step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)
    def forward(self, x):
        # x: [T, B, C]
        B, T, C = x.size()
        fold = C // self.fold_div # C = fold * self.fold_div
        out = torch.zeros_like(x)

        # V1
        # 后移：第 1 段
        # out[:, 1:, :fold] = x[:, :-1, :fold] + x[:, 1:, :fold]
        out[:, 1:, :fold] = x[:, :-1, :fold]

        # 不动：中间部分
        out[:, :, fold:(self.fold_div - 1) * fold] = x[:, :, fold:(self.fold_div - 1) * fold]

        # 前移：最后一段
        # out[:, :-1, (self.fold_div - 1) * fold:] = x[:, 1:, (self.fold_div - 1) * fold:] + x[:, :-1, (self.fold_div - 1) * fold:]
        out[:, :-1, (self.fold_div - 1) * fold:] = x[:, 1:, (self.fold_div - 1) * fold:]

        # # V2
        # # 后移：第 1 段
        # # out[:, 1:, :fold] = x[:, :-1, :fold] + x[:, 1:, :fold]
        # out[:, 1:, :fold] = x[:, :-1, :fold]
        #
        # # 前移：第 2 段
        # # out[:, :, fold:(self.fold_div - 1) * fold] = x[:, :, fold:(self.fold_div - 1) * fold]
        # out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold]
        #
        # # 剩下不动
        # # out[:, :-1, (self.fold_div - 1) * fold:] = x[:, 1:, (self.fold_div - 1) * fold:] + x[:, :-1, (self.fold_div - 1) * fold:]
        # out[:, :-1, 2*fold:] = x[:, 1:, 2 * fold:]
        #
        # # out = self.lif(out)

        return out

class SpikingEmbedv4(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv4, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True)
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                                  bias=config.use_dw_bias)

        self.linear = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=False, step_mode='m') # self.config.bias

        if self.config.use_bn:
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)

        if self.config.spike_mode == 'lif':
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

        elif self.config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

            self.lif2 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold, v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend
            )

        self.temporal_shift = TemporalShift1D(fold_div=8)

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
        x = self.temporal_shift(x)
        x = x + x_res


        x = self.linear(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)

        return x



class SpikeDrivenTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # self.spike_embed = SpikingEmbed(config)
        # self.spike_embed = SpikingEmbedv2(config, kernel_size=7) # 相较于V1 添加了一个linear层
        self.spike_embed = SpikingEmbedv2p(config, kernel_size=7) # 相较于V1 添加了一个linear层, 目前最好
        # self.spike_embed = SpikingEmbedv2pgated(config, kernel_size=7) # 相较于V1 添加了一个linear层, 目前有望最好
        # self.spike_embed = SpikingEmbedv2pd(config, kernel_size=5) # 相较于V1 添加了一个linear层
        # self.spike_embed = SpikingEmbedv2pcs(config, kernel_size=7) # 相较于V1 添加了一个linear层
        # self.spike_embed = SpikingEmbedv2ptsm(config, kernel_size=7) # 相较于V1 添加了一个linear层
        # self.spike_embed = SpikingEmbedv2pgru(config, kernel_size=7) # 相较于V1 添加了一个linear层
        # self.spike_embed = SpikingEmbedv2pp(config, kernel_size=7) # 相较于V1 添加了一个linear层
        # self.spike_embed = SpikingEmbedv3(config) # 加入了Spatial-Temporal  Interaction
        # self.spike_embed = SpikingEmbedv4(config) # 在linear前加入了Temporal shift

        self.blocks = nn.ModuleList(
            [
                MS_Block_Conv(
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
        # print(self.blocks)

        if self.config.use_dp:
            self.final_dp = layer.Dropout(self.config.dropout_l, step_mode='m')

        self.head = layer.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=False, step_mode='m')
        # self.vote_head = layer.VotingLayer(self.config.n_outputs, step_mode='m')

        # if self.config.spike_mode == 'lif':
        #     self.final_lif = LIFNode(tau=self.config.init_tau, v_threshold=1e9,
        #                              surrogate_function=self.config.surrogate_function,
        #                              detach_reset=self.config.detach_reset,
        #                              step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)

        self._reset_parameters()


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x, attention_mask):

        x = self.spike_embed(x)

        # block_start_time = time.time()
        # Iterate over each module in blocks and apply it to x
        for module in self.blocks:
            x = module(x, attention_mask)

        if self.config.use_dp:
            x = self.head(self.final_dp(x))
        else:
            x = self.head(x)

        # x = self.final_lif(x)

        return x


# # 示例配置对象（请确保字段与模型定义中一致）
# class Config:
#     n_inputs = 140              # 输入特征维度 D
#     n_hidden_neurons = 256      # 隐藏神经元数（可改为你模型需要的值）
#     n_outputs = 35              # 输出类别数（根据任务调整）
#     init_tau = 2.0
#     v_threshold = 1.0
#     surrogate_function = neuron.surrogate.Sigmoid()
#     detach_reset = True
#     spike_mode = 'lif'          # 或 'plif'
#     dropout_l = 0.1
#     backend = 'torch'
#     use_dp = True
#     use_norm = True
#     depths = 4                  # Transformer Block 数
#     num_heads = 4               # 多头注意力头数
#     bias = True                 # Conv1D 是否带 bias
#     use_ln = False
#     dropout_p = 0.1
#     mlp_ratio = 4
#     split_ratio = 1
#     hidden_dims = mlp_ratio*n_hidden_neurons # 可以试一下768,
#     kernel_size = 31  # 卷积核为255时，92.42% 255=>127=>63=>31,
#     use_dw_bias = False
#     use_bn = True
#     attn_mode = 'v2'
#     backend = 'cupy'
#
# seed_val = 42
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# device = torch.device(3)
# # 初始化配置和模型
# config = Config()
# model = SpikeDrivenTransformer(config).to(device)
# # 生成一个长度为100的tensor，每个值为100
# x_len = torch.full((32,), 100)  # 等价于 torch.ones(100) * 100
# # 构造输入张量：shape 为 (Batch, Time, D)
# x = torch.randn(32, 100, 140).to(device)  # bs=32, t=100, D=140
# attention_mask = padded_sequence_mask(x_len)
# attention_mask = attention_mask.transpose(0, 1).to(device)
# # attention mask 可以为 None 或构造一个合适的 mask
# # 例如这里假设不使用 attention mask，可设置为 None 或全1 mask
# # attention_mask = None
#
# # 模型前向传播
# with torch.no_grad():
#     output = model(x, attention_mask)  # shape: (Batch, Time, n_outputs)
#
# print('Output shape:', output.shape)
# print(output)