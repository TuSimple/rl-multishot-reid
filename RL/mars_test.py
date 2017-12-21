import sys
#sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx
import numpy as np
import argparse
import random


from symbols import sym_base_net, sym_DQN
from utils import get_imRecordIter, load_checkpoint


cmcs = [1, 5, 10, 20]
cn = 4


def norm_cnts(cnts, cnt):
    return [cnts[i] / cnt[0] for i in xrange(cn)]


def update_cnts(cur, cnts, cnt):
    for j in xrange(cn):
        if cur < cmcs[j]:
            cnts[j] += 1.0
    cnt[0] += 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='single domain car recog training')
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1"')
    parser.add_argument('--model-load-epoch', type=int, default=3,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default='mars-TEST-DQN_test-2017.11.15-10.51.56-bs4-ss8-incp_prep__nobg_noregQv_block2_f2_nofus0-2_poscontra_fne0.1-1-1_tisr1-sgd_t500-_qg0.9-up0.2-vtd4.0-_lr1e1-_32-1024-_na3-3',
                        help='load model prefix')
    return parser.parse_args()


def create_module(ctx, seq_len, is_train=False):
    net = sym_DQN(args, is_train=is_train, num_acts=args.num_acts, bn=False, global_stats=False, no_his=True)
    mod = mx.mod.Module(symbol=net, data_names=('data1', 'data2'), label_names=None,
                        fixed_param_names=['data1', 'data2'], context=ctx)
    mod.bind(data_shapes=[('data1', (seq_len, 1024)), ('data2', (seq_len, 1024)),],
             for_training=is_train, inputs_need_grad=False)
    return mod


def dist(a, b):
    diff = a - b
    return mx.nd.sum(diff*diff).asnumpy()[0]


def copyto(x):
    return x.copyto(x.context)


def get_train_args(name):
    fn = open('log/%s.log'%name)
    s = fn.readline()[10:]
    fn.close()
    s = 'ret=argparse.' + s
    exec(s)
    return ret

test_args = parse_args()
print 'test arg:', test_args
devices = [mx.gpu(int(i)) for i in test_args.gpus.split(',')]

model_path = 'models'

args = get_train_args(test_args.model_load_prefix)
print 'train arg:', args

model_path = 'models'
arg_params, aux_params = load_checkpoint(
    '%s/%s' % (model_path, test_args.model_load_prefix), test_args.model_load_epoch)
base_mod = mx.mod.Module(symbol=sym_base_net(args.network, is_test=True), data_names=('data',), label_names=None, context=devices)
base_mod.bind(data_shapes=[('data', (1024, 3, 224, 112))], for_training=False)
base_mod.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)

dataiter = get_imRecordIter(
    args, 'recs/eval_test', (3, 224, 112), 1024,
    shuffle=False, aug=False, even_iter=True)
dataiter.reset()
F = base_mod.predict(dataiter)
del dataiter
print 'l2'
print F

print 'base feat predicted'
query = np.loadtxt('/data3/matt/MARS/MARS-evaluation/info/query.csv', delimiter=',').astype(int)
gallery = np.loadtxt('/data3/matt/MARS/MARS-evaluation/info/gallery.csv', delimiter=',').astype(int)

cnts, cnt = [0, 0, 0, 0], [0]
max_turn, tot_ava = args.sample_size, 1

query[:, 2] -= 1
gallery[:, 0] -= 1



def get_data(a):
    return F[random.randrange(gallery[a, 0], gallery[a, 1])]

P = mx.nd.zeros((gallery.shape[0], args.sample_size, F.shape[1]),ctx=devices[0])
for a in xrange(gallery.shape[0]):
    j, camj = gallery[a, 2:]
    if j == -1 or gallery[a, 0] == gallery[a, 1]:
        continue
    cur = mx.nd.zeros((args.sample_size, F.shape[1]),ctx=devices[0])
    for k in xrange(args.sample_size):
        cur[k] = get_data(a)
    P[a] = copyto(cur)

data1 = mx.symbol.Variable(name="data1")
data2 = mx.symbol.Variable(name="data2")
Qsym = sym_DQN(data1, data2, args.num_sim, args.num_hidden, is_train=False, num_acts=args.num_acts, min_states=args.min_states, min_imgs=args.min_imgs, fusion=args.fusion, bn=args.q_bn, global_stats=False, no_his=False)
Q = mx.mod.Module(symbol=Qsym, data_names=('data1', 'data2'), label_names=None, context=devices[0])
Q.bind(data_shapes=[('data1', (args.sample_size, F.shape[1])),
                    ('data2', (args.sample_size, F.shape[1]))],
                    for_training=False)
Q.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True, allow_missing=False)
cnts, cnt = [0, 0, 0, 0], [0]
pc, ps = 0, 0
hists = [0 for _ in xrange(args.sample_size)]
for q in xrange(query.shape[0]):
    i, cam, idx = query[q]
    if gallery[idx, 0] == gallery[idx, 1]:
        continue
    d = []
    for a in xrange(gallery.shape[0]):
        j, camj = gallery[a, 2:]
        if j == i and camj == cam or j == -1 or gallery[a, 0] == gallery[a, 1]:
            continue
        if random.random() > 0.01:
            continue
        Q.forward(mx.io.DataBatch([P[idx], P[a]], []), is_train=False)
        Qvalues = Q.get_outputs()[0].asnumpy()
        for k in xrange(args.sample_size):
            if Qvalues[k, 2] < Qvalues[k, 0] or Qvalues[k, 2] < Qvalues[k, 1] or k == args.sample_size - 1:
                d.append((Qvalues[k, 0] - Qvalues[k, 1], j))
                ps += k + 1
                pc += 1
                hists[k] += 1
                break

    d = sorted(d)
    cur = 0
    for a in xrange(len(d)):
        j = d[a][1]
        if j == i:
            break
        else:
            cur += 1
    update_cnts(cur, cnts, cnt)
    print q, i, cam, idx, cur, norm_cnts(cnts, cnt), ps * 1.0 / pc