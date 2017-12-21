import find_mxnet
import mxnet as mx
import numpy as np


def get_symbol(params=None):
    if params is None:
        params = dict([(name, mx.sym.Variable(name)) for name in\
                ['conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias',
                 'conv3_weight', 'conv3_bias', 'conv4_weight', 'conv4_bias',
                 'conv5_weight', 'conv5_bias']
        ])

    # data
    x = mx.symbol.Variable(name="data")
    x = mx.sym.Convolution(data=x, kernel=(11, 11), stride=(4, 4), num_filter=96, weight=params['conv1_weight'], bias=params['conv1_bias'])
    x = mx.sym.Activation(data=x, act_type='relu')
    x = mx.sym.LRN(data=x, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    x = mx.sym.Pooling(data=x, pool_type='max', kernel=(3, 3), stride=(2, 2))

    x = mx.sym.Convolution(data=x, kernel=(5, 5), pad=(2, 2), num_filter=256, num_group=2, weight=params['conv2_weight'], bias=params['conv2_bias'])
    x = mx.sym.Activation(data=x, act_type='relu')
    x = mx.sym.LRN(data=x, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pool_type='max')

    x = mx.sym.Convolution(data=x, kernel=(3, 3), pad=(1, 1), num_filter=384, num_group=1, weight=params['conv3_weight'], bias=params['conv3_bias'])
    x = mx.sym.Activation(data=x, act_type='relu')

    x = mx.sym.Convolution(data=x, kernel=(3, 3), pad=(1, 1), num_filter=384, num_group=2, weight=params['conv4_weight'], bias=params['conv4_bias'])
    x = mx.sym.Activation(data=x, act_type='relu')

    x = mx.sym.Convolution(data=x, kernel=(3, 3), pad=(1, 1), num_filter=256, num_group=2, weight=params['conv5_weight'], bias=params['conv5_bias'])
    x = mx.sym.Activation(data=x, act_type='relu')
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pool_type='max')

    return x
