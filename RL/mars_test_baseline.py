import sys
#sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx

import numpy as np
import argparse
import random

from symbols import sym_base_net, sym_DQN
from utils import get_imRecordIter, load_checkpoint
from sklearn.metrics import average_precision_score


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
    parser.add_argument('--gpus', type=str, default='2',
                        help='the gpus will be used, e.g "0,1"')
    parser.add_argument('--data-dir', type=str,
                        default="/data3/matt/MARS",
                        help='data directory')
    parser.add_argument('--sample-size', type=int, default=8,
                        help='sample frames from each video')
    parser.add_argument('--base-model-load-epoch', type=int, default=1,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--base-model-load-prefix', type=str, default='mars_alex',#'mars_baseline_b8',#
                        help='load model prefix')
    parser.add_argument('--network', type=str,
                        default='alexnet',#'inception-bn',#
                        help='network name')
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



args = parse_args()
print args
devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]

batch_size = 128
seq_len = batch_size
model_path = 'models'
random.seed(19930214)
pool = set([random.randrange(1980) for _ in xrange(100)])
print pool

arg_params, aux_params = load_checkpoint(
    '../baseline/%s/%s' % (model_path, args.base_model_load_prefix), args.base_model_load_epoch)
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

max_turn, gc = 10, 0
max_frames = max_turn
cmc1 = np.zeros(max_frames)
MAP = np.zeros(max_frames)
results = []

def dist(a, b):
    diff = a - b
    return mx.nd.sum(diff*diff).asnumpy()[0]

cur = mx.nd.zeros((gallery.shape[0], F.shape[1]),ctx=devices[0])
avgs = mx.nd.zeros((max_frames, gallery.shape[0], F.shape[1]),ctx=devices[0])
for k in xrange(max_frames):
    for a in xrange(gallery.shape[0]):
        j, camj = gallery[a, 2:]
        if j == -1 or gallery[a, 0] == gallery[a, 1]:
            continue
        cur[a] = get_data(a)
    if k == 0:
        avgs[k] = copyto(cur)
    else:
        avgs[k] = (avgs[k - 1] + cur)
for k in xrange(1, max_frames):
    avgs[k] /= (k + 1)


cnts_g, cnt_g = [0, 0, 0, 0], [0]

for q in xrange(query.shape[0]):
    i, cam, idx = query[q]
    if gallery[idx, 0] == gallery[idx, 1]:
        continue
    scores = []
    labels = []
    d = []
    for k in xrange(max_frames):
        g = avgs[k]
        diff = g[idx] - g
        d.append(mx.nd.sum(diff*diff, axis=1).asnumpy())
    for a in xrange(gallery.shape[0]):
        j, camj = gallery[a, 2:]
        if j == i and camj == cam or j == -1 or gallery[a, 0] == gallery[a, 1]:
            continue
        g = []
        for k in xrange(max_frames):
            g.append(d[k][a])
        scores.append(g)
        labels.append(1 if i == j else 0)

    scores = np.array(scores)
    for j in xrange(max_frames):
        MAP[j] += average_precision_score(labels, -scores[:, j])
        a = np.argmin(scores[:, j])
        if labels[a] == 1:
            cmc1[j] += 1
    gc += 1
    print q, i, cam, idx, gc
    print MAP[:10] / gc
    print cmc1[:10] / gc

'''
avgs = []
for a in xrange(gallery.shape[0]):
    j, camj = gallery[a, 2:]
    if j == -1 or gallery[a, 0] == gallery[a, 1]:
        avgs.append(None)
        continue
    for k in xrange(gallery[a, 0], gallery[a, 1]):
        if k == gallery[a, 0]:
            avg_opp = copyto(F[k])
        else:
            avg_opp += F[k]
    avg_opp /= gallery[a, 1] - gallery[a, 0]
    avgs.append(avg_opp)

for q in xrange(query.shape[0]):
    if not q in pool:
        continue
    i, cam, idx = query[q]
    if gallery[idx, 0] == gallery[idx, 1]:
        continue
    scores = []
    label = []
    d = []
    for k in xrange(gallery[idx, 0], gallery[idx, 1]):
        if k == gallery[idx, 0]:
            avg_cur = copyto(F[k])
        else:
            avg_cur += F[k]
    avg_cur /= gallery[idx, 1] - gallery[idx, 0]
    for a in xrange(gallery.shape[0]):
        j, camj = gallery[a, 2:]
        if j == i and camj == cam or j == -1 or gallery[a, 0] == gallery[a, 1]:
            continue
        d.append((dist(avg_cur, avgs[a]), j))

    d = sorted(d)
    cur = 0
    for a in xrange(len(d)):
        j = d[a][1]
        if j == i:
            break
        else:
            cur += 1
    update_cnts(cur, cnts, cnt)
    print q, i, cam, idx, cur, norm_cnts(cnts, cnt)
'''
