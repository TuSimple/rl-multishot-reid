import sys
#sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx
from mxnet.optimizer import SGD, Adam, RMSProp

import numpy as np

from symbols import sym_base_net, sym_DQN
from utils import load_checkpoint, TimeInvScheduler, dist, copyto
from base_module import BaseModule
import os

def create_moduleQ(data1, data2, ctx, seq_len, num_sim, num_hidden, num_acts, min_states, min_imgs, fusion=False, bn=False, is_train=False, nh=False, is_e2e=False):
    os.environ['MXNET_EXEC_INPLACE_GRAD_SUM_CAP'] = str(100)
    net = sym_DQN(data1, data2, num_sim, num_hidden, is_train=is_train, num_acts=num_acts, min_states=min_states, min_imgs=min_imgs, fusion=fusion, bn=bn, global_stats=False, no_his=False)
    mod = BaseModule(symbol=net, data_names=('data1', 'data2'), label_names=None,
                     fixed_param_names=[] if is_e2e else ['data1', 'data2'], context=ctx)
    mod.bind(data_shapes=[('data1', (seq_len, 3, 224, 112)),
                          ('data2', (seq_len, 3, 224, 112))],
             for_training=is_train, inputs_need_grad=False)
    return mod


def get_optimizer(args):
    assert args.optimizer in ['sgd', 'adam', 'rms']

    if args.optimizer == 'sgd':
        stepPerEpoch = args.num_examples
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[stepPerEpoch * int(x) for x in args.lr_step.split(',')], factor=0.1)
        #lr_scheduler = TimeInvScheduler(step=args.tisr) # Time inverse scheduler
        return SGD(learning_rate=args.lr, momentum=0.9,
                   wd=0.0001, clip_gradient=10, lr_scheduler=lr_scheduler,
                   rescale_grad=1.0)
    elif args.optimizer == 'rms':
        return RMSProp(learning_rate=args.lr, wd=0.0001)
    else:
        return Adam(learning_rate=args.lr, wd=0.0001, clip_gradient=10)


def get_Qvalue(Q, data, is_train=False):
    data_batch = mx.io.DataBatch([data[0], data[1]], [])
    Q.forward(data_batch, is_train=is_train)
    return Q.get_outputs()[0].asnumpy()


def wash(data, ctx):
    ret = []
    if isinstance(data[0], list):
        for i in xrange(len(data[0])):
            t = []
            for j in xrange(len(data)):
                t.append(np.expand_dims(data[j][i], axis=0) if data[j][i].shape[0] > 1 or len(data[j][i].shape) == 1 else data[j][i])
            t = np.concatenate(t)
            ret.append(mx.nd.array(t, ctx=ctx))
    else:
        for i in xrange(len(data)):
            ret.append(mx.nd.array(data[i], ctx=ctx))
    return ret


class Agent:
    def __init__(self, args, devices):
        self.e2e = args.e2e
        self.his = args.history
        arg_params, aux_params = load_checkpoint('../baseline/models/%s' % args.model_load_prefix, args.model_load_epoch)
        data1, data2 = sym_base_net(args.network, is_train=args.e2e, global_stats=True)
        init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
        opt = get_optimizer(args)
        self.Q = create_moduleQ(data1, data2, devices, args.sample_size, args.num_sim, args.num_hidden, args.num_acts, args.min_states, args.min_imgs, fusion=args.fusion, is_train=True, nh=not args.history, is_e2e=args.e2e, bn=args.q_bn)
        self.Q.init_params(initializer=init,
                           arg_params=arg_params,
                           aux_params=aux_params,
                           allow_missing=True,
                           force_init=True)
        self.Q.init_optimizer(optimizer=opt)
        self.target_cnt = 1
        self.devices = devices
        self.prefix = 'models/%s' % args.mode
        self.batch_size = args.batch_size
        self.update_cnt = 0
        self.Q.clear_gradients()
        self.gradQ = [[grad.copyto(grad.context) for grad in grads] for grads in self.Q._exec_group.grad_arrays]

    def wash_data(self, data):
        return wash(data, self.devices)

    def get_Qvalue(self, data, is_train=False):
        return get_Qvalue(self.Q, data, is_train=is_train)

    def update(self, grad):
        self.Q.backward(grad)
        for gradsr, gradsf in zip(self.Q._exec_group.grad_arrays, self.gradQ):
            for gradr, gradf in zip(gradsr, gradsf):
                gradf += gradr
        self.Q.clear_gradients()
        self.update_cnt += 1
        if self.update_cnt % self.batch_size == 0:
            print 'update', self.update_cnt
            for gradsr, gradsf in zip(self.Q._exec_group.grad_arrays, self.gradQ):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr[:] = gradf[:] / self.batch_size
            self.Q.update()
            for grads in self.gradQ:
                for grad in grads:
                    grad[:] = 0

    def save(self, e):
        self.Q.save_params('%s-%04d.params'%(self.prefix, e))
