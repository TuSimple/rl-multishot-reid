from collections import namedtuple
import find_mxnet
import mxnet as mx

LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])

def get_lstm_cell(i):
    return LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i))

def lstm(num_hidden, indata, prev_h, prev_c, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return next_h, next_c


GRUParam = namedtuple("GRUParam", ["gates_i2h_weight", "gates_i2h_bias",
                                   "gates_h2h_weight", "gates_h2h_bias",
                                   "trans_i2h_weight", "trans_i2h_bias",
                                   "trans_h2h_weight", "trans_h2h_bias"])

def get_gru_cell(i):
    return GRUParam(gates_i2h_weight=mx.sym.Variable("l%d_i2h_gates_weight" % i),
                                    gates_i2h_bias=mx.sym.Variable("l%d_i2h_gates_bias" % i),
                                    gates_h2h_weight=mx.sym.Variable("l%d_h2h_gates_weight" % i),
                                    gates_h2h_bias=mx.sym.Variable("l%d_h2h_gates_bias" % i),
                                    trans_i2h_weight=mx.sym.Variable("l%d_i2h_trans_weight" % i),
                                    trans_i2h_bias=mx.sym.Variable("l%d_i2h_trans_bias" % i),
                                    trans_h2h_weight=mx.sym.Variable("l%d_h2h_trans_weight" % i),
                                    trans_h2h_bias=mx.sym.Variable("l%d_h2h_trans_bias" % i))

def gru(num_hidden, indata, prev_h, param, seqidx, layeridx, dropout=0.):
    """
    GRU Cell symbol
    Reference:
    * Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural
        networks on sequence modeling." arXiv preprint arXiv:1412.3555 (2014).
    """
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.gates_i2h_weight,
                                bias=param.gates_i2h_bias,
                                num_hidden=num_hidden * 2,
                                name="t%d_l%d_gates_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_h,
                                weight=param.gates_h2h_weight,
                                bias=param.gates_h2h_bias,
                                num_hidden=num_hidden * 2,
                                name="t%d_l%d_gates_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=2,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    update_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    reset_gate = mx.sym.Activation(slice_gates[1], act_type="sigmoid")
    # The transform part of GRU is a little magic
    htrans_i2h = mx.sym.FullyConnected(data=indata,
                                       weight=param.trans_i2h_weight,
                                       bias=param.trans_i2h_bias,
                                       num_hidden=num_hidden,
                                       name="t%d_l%d_trans_i2h" % (seqidx, layeridx))
    h_after_reset = prev_h * reset_gate
    htrans_h2h = mx.sym.FullyConnected(data=h_after_reset,
                                       weight=param.trans_h2h_weight,
                                       bias=param.trans_h2h_bias,
                                       num_hidden=num_hidden,
                                       name="t%d_l%d_trans_i2h" % (seqidx, layeridx))
    h_trans = htrans_i2h + htrans_h2h
    h_trans_active = mx.sym.Activation(h_trans, act_type="tanh")
    next_h = prev_h + update_gate * (h_trans_active - prev_h)
    return next_h
