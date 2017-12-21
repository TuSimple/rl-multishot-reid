import sys
#sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx
import numpy as np
import argparse
import random
import importlib

from utils import get_imRecordIter, load_checkpoint
from sklearn.metrics import average_precision_score


cmcs = [1, 5, 10, 20]
cn = 4


def norm_cnts(cnts, cnt):
    return [cnts[i] / cnt[0] for i in xrange(cn)]


def update_cnts(d, cnts, cnt, N, i):
    #r = np.argsort(d)
    dcur = d[i]
    cur, pre = 0, []
    for j in xrange(N):
        if d[j] <= d[i] and not j == i:
            cur += 1
            pre.append(j)
    for j in xrange(cn):
        if cur < cmcs[j]:
            cnts[j] += 1.0
    cnt[0] += 1.0
    return pre


def parse_args():
    parser = argparse.ArgumentParser(
        description='single domain car recog training')
    parser.add_argument('--gpus', type=str, default='6',
                        help='the gpus will be used, e.g "0,1"')
    parser.add_argument('--data-dir', type=str,
                        default='/data3/matt/iLIDS-VID/recs',#"/data3/matt/prid_2011/recs",#
                        help='data directory')
    parser.add_argument('--num-examples', type=int, default=10000,
                        help='the number of training examples')
    parser.add_argument('--num-id', type=int, default=100,
                        help='the number of training ids')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--base-model-load-epoch', type=int, default=1,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--base-model-load-prefix', type=str, default='ilds_baseline',
                        help='load model prefix')
    parser.add_argument('--dataset', type=str, default='image_test',
                        help='dataset (test/query)')
    parser.add_argument('--network', type=str,
                        #default='alexnet', help='network name')
                        default='inception-bn', help='network name')
    return parser.parse_args()


def build_base_net(args, is_train=False, global_stats=False):
    '''
    network structure
    '''
    symbol = importlib.import_module('symbol_' + args.network).get_symbol()
    # concat = internals["ch_concat_5b_chconcat_output"]
    #symbol = mx.symbol.Dropout(data=symbol, name='dropout1')
    pooling = mx.symbol.Pooling(
        data=symbol, kernel=(1, 1), global_pool=True,
        pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')
    l2 = mx.symbol.L2Normalization(data=flatten, name='l2_norm')
    return l2

args = parse_args()
print args
devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]
batch_size = args.batch_size
model_path = 'models'

def dist(a, b):
    diff = a - b
    return mx.nd.sum(diff*diff).asnumpy()[0]

def copyto(x):
    return x.copyto(x.context)

cnts_g, cnt_g = [0, 0, 0, 0], [0]
max_turn, gc = 10, 0
max_frames = max_turn
cmc1 = np.zeros(max_frames)
MAP = np.zeros(max_frames)
results = []
for sets in xrange(10):
    arg_params, aux_params = load_checkpoint(
        '%s/%s_%d' % (model_path, args.base_model_load_prefix, sets), args.base_model_load_epoch)
    base_mod = mx.mod.Module(symbol=build_base_net(args), data_names=('data', ), label_names=None, context=devices)
    base_mod.bind(data_shapes=[('data', (args.batch_size, 3, 224, 112))], for_training=False)
    base_mod.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)

    dataiter = get_imRecordIter(
        args, '%s%d' % (args.dataset, sets), (3, 224, 112), args.batch_size,
        shuffle=False, aug=False, even_iter=False)

    dataiter.reset()

    output = base_mod.predict(dataiter)
    F = output
    F2 = F
    print F.shape

    cnt_lst = np.loadtxt(args.data_dir + '/' + 'image_test' + str(sets) + '.txt').astype(int)
    N = cnt_lst.shape[0] / 2

    avp = []
    for i in xrange(N + N):
        for j in xrange(cnt_lst[i], cnt_lst[i + 1]):
            if j == cnt_lst[i]:
                g = copyto(F[j])
            else:
                g += F[j]
        avp.append(g / mx.nd.sqrt(mx.nd.sum(g * g)))

    cnts, cnt = [0, 0, 0, 0], [0]

    for i in xrange(N+N):
        d = []
        a = i % N
        scores = []
        label = np.array([(1 if _ == a else 0) for _ in xrange(N)])
        for j in xrange(N):
            d.append(dist(avp[i], avp[j if i >= N else (j + N)]))
            g, x, y = [], mx.nd.zeros((int(F.shape[1])),ctx=devices[0]), mx.nd.zeros((int(F.shape[1])),ctx=devices[0])
            for k in xrange(max_frames):
                if i < N:
                    x += F2[random.randrange(cnt_lst[i], cnt_lst[i+1])]
                    y += F2[random.randrange(cnt_lst[j+N], cnt_lst[j+1+N])]
                else:
                    x += F2[random.randrange(cnt_lst[i], cnt_lst[i+1])]
                    y += F2[random.randrange(cnt_lst[j], cnt_lst[j+1])]
                g.append(dist(x/(k+1), y/(k+1)))
            scores.append(g)
        scores = np.array(scores)
        for j in xrange(max_frames):
            MAP[j] += average_precision_score(label, -scores[:, j])
            if min(scores[:, j]) == scores[a, j]:
                cmc1[j] += 1
        gc += 1
        print gc
        print MAP[:10] / gc
        print cmc1[:10] / gc
        update_cnts(d, cnts, cnt, N, i if i < N else i - N)
        update_cnts(d, cnts_g, cnt_g, N, i if i < N else i - N)
        print i, norm_cnts(cnts, cnt), norm_cnts(cnts_g, cnt_g)
    results.append((cnts, cnt))