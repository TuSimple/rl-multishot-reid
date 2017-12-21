import sys
#sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx

import numpy as np
import argparse
from sklearn.metrics import average_precision_score

from batch_provider import BatchProvider
from utils import get_imRecordIter, load_checkpoint
from agent import sym_base_net, wash, get_Qvalue, create_moduleQ

import cv2, os

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 24
mpl.rcParams['font.family'] = "Times New Roman"
mpl.rcParams['legend.fontsize'] = "small"
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['figure.figsize'] = 9, 6.3
mpl.rcParams['legend.labelspacing'] = 0.1
mpl.rcParams['legend.borderpad'] = 0.1
mpl.rcParams['legend.borderaxespad'] = 0.2
mpl.rcParams['font.monospace'] = "Courier 10 Pitch"
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']


def plot(Q, terminal, name):
    t = [1+_ for _ in xrange(terminal)]
    a = [Q[_, 1] for _ in xrange(terminal)]
    b = [Q[_, 0] for _ in xrange(terminal)]
    c = [Q[_, 2] for _ in xrange(terminal)]

    plt.figure(figsize=(10,9))
    ax = plt.gca()

    plt.plot(t, a,
             marker='o',
             markersize=12,
             markerfacecolor=(0, 1, 0, 0.5),
             color='g',
             label='same',
             alpha=0.5,
             )

    plt.plot(t, b,
             marker='x',
             markersize=12,
             markerfacecolor=(1, 0, 0, 0.5),
             color='r',
             label='different',
             alpha=0.5,
             )

    plt.plot(t, c,
             marker='^',
             markersize=12,
             markerfacecolor=(1, 1, 0, 0.5),
             color='y',
             label='unsure',
             alpha=0.5,
             )

    plt.ylabel(r'\textbf{Q-Value}')
    plt.xlabel(r'\textbf{\#. Time Steps}')
    plt.grid(linestyle=':')
    plt.savefig('%s.pdf'%name)


def parse_args():
    parser = argparse.ArgumentParser(
        description='single domain car recog training')
    parser.add_argument('--gpus', type=str, default='2',
                        help='the gpus will be used, e.g "0,1"')
    parser.add_argument('--model-load-epoch', type=int, default=3,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default='ilds-TEST-DQN_test-1-2017.11.14-23.56.43-bs4-ss8-incp_prep__nobg_noregQv_block2_f2_nofus0-2_poscontra_fne0.1-1-1_tisr1-sgd_t500-_qg0.9-up0.2-vtd4.0-_lr1e1-_32-1024-_na3-3',
                        help='load model prefix')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='the batch size')
    parser.add_argument('--boost-times', type=int, default=1,
                        help='boosting times to increase robustness')
    return parser.parse_args()


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
args = get_train_args(test_args.model_load_prefix)
print 'train arg:', args

batch_size = args.batch_size
num_epoch = args.num_epoches

arg_params, aux_params = load_checkpoint('models/%s' % test_args.model_load_prefix, test_args.model_load_epoch)
data1, data2 = sym_base_net(args.network, is_train=args.e2e, global_stats=True)
Q = create_moduleQ(data1, data2, devices, args.sample_size, args.num_sim, args.num_hidden, args.num_acts, args.min_states, args.min_imgs, fusion=args.fusion, is_train=True, nh=not args.history, is_e2e=args.e2e, bn=args.q_bn)
Q.init_params(initializer=None,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=False,
              force_init=True)


valid_iter = get_imRecordIter(
           args, 'recs/%s'%args.valid_set, (3, 224, 112), 1,
           shuffle=False, aug=False, even_iter=True)
train_iter = get_imRecordIter(
           args, 'recs/%s'%args.train_set, (3, 224, 112), 1,
           shuffle=False, aug=True, even_iter=True)

valid_lst = np.loadtxt('%s/recs/%s.txt'%(args.data_dir, args.valid_set)).astype(int)
train_lst = np.loadtxt('%s/recs/%s.txt'%(args.data_dir, args.train_set)).astype(int)

valid = BatchProvider(valid_iter, valid_lst, False, args.sample_size, sample_ratio=0.5, is_valid=True, need_feat=args.history)
train = BatchProvider(train_iter, train_lst, True, args.sample_size, sample_ratio=0.5, need_feat=args.history)
N = args.num_id

cmcs, ap, cmcn, vscores, vturns = [[], [], [], []], [], [1, 5, 10, 20], [], []
max_penalty=1


def tocv2(im):
    newim = np.zeros_like(im)
    newim[0] = im[2]
    newim[1] = im[1]
    newim[2] = im[0]
    newim = np.transpose(newim, (1, 2, 0)) + 128
    newim = newim.astype(np.uint8)
    print newim
    return newim

valid.reset()
batch, valid_cnt, vv, vpool = 0, 0, np.zeros((N*2, N)), set()
vs, vt = [0 for i in xrange(N+N)], [0 for i in xrange(N+N)]
fts = [[0 for _2 in xrange(N)] for _1 in xrange(N)]
fdir = 'plots/%s'%args.mode
os.system('mkdir %s'%fdir)
for i in xrange(args.sample_size + 1):
    os.system('mkdir %s/%d'%(fdir,i))
    for j in xrange(4):
        os.system('mkdir %s/%d/%d'%(fdir,i,j))

while valid_cnt < N*N:
    batch += 1
    cur, a, b = valid.provide()
    y = ((a %N) == (b % N))
    data_batch = wash(cur, devices[0])
    Qvalue = get_Qvalue(Q, data_batch, is_train=False)
    print Q.get_outputs()[1].asnumpy()
    print Qvalue
    i = 0
    while i < args.sample_size:
        if args.total_forward:
            if i + 1 < args.sample_size:
                k = 2
            else:
                k = np.argmax(Qvalue[i, :2])
        else:
            k = np.argmax(Qvalue[i])
        cls = k % args.acts_per_round
        step = k - 1
        if cls >= 2:
            if i + step >= args.sample_size:
                r = -max_penalty
                terminal = True
            else:
                r = -args.penalty * (2.0 - (0.5 ** (step - 1)))
                terminal = False
        else:
            r = 1 if cls == y else -max_penalty
            terminal = True
        if args.pos_weight > 1:
            if y:
                r *= args.pos_weight
        else:
            if not y:
                r /= args.pos_weight
        print 'valid', i, (a, b), Qvalue[i], k, (y, cls), r
        va, vb = a, b % N
        if (va, vb) not in vpool:
            vs[va] += r
            vs[vb+N] += r
        if terminal:
            if (va, vb) not in vpool:
                fts[va][vb] = (k + (3 if va == vb else 0), i)
                vpool.add((va, vb))
                valid_cnt += 1
                vv[va][vb] = Qvalue[i][0] - Qvalue[i][1]
                vt[va] += i + 1
                vv[vb+N][va] += vv[va][vb]
                vt[vb+N] += i + 1
                print va, vb, vv[va][vb], r
        if terminal and r == 1 and i > 0:
            img = np.zeros((3, cur[0][0].shape[1] * 2, cur[0][0].shape[2] * args.sample_size))
            print va, vb, i
            for j in xrange(args.sample_size):
                img[:, :cur[0][j].shape[1], cur[0][j].shape[2]*j:cur[0][j].shape[2]*(j+1)] = cur[0][j]
                img[:, cur[0][j].shape[1]:, cur[1][j].shape[2]*j:cur[1][j].shape[2]*(j+1)] = cur[1][j]
            name = '%s/%d/%d/%d-%d'%(fdir, i if cls < 2 else args.sample_size, ((2 if y else 0) + (1 if Qvalue[i, 1] > Qvalue[i, 0] else 0)), va, vb)
            cv2.imwrite('%s.png'%(name), tocv2(img))
            np.savetxt('%s.txt'%name, Qvalue)
            plot(Qvalue, i + 1, name)
        if terminal:
            break
        i += step
for i in xrange(N*2):
    a, r = i % N, 0
    for b in xrange(N):
        if a != b and vv[i][b] <= vv[i][a]:
            r += 1
    for k in xrange(4):
        cmcs[k].append(1.0 if r < cmcn[k] else 0.0)
    vscores += [vs[i]]
    vturns += [vt[i]]
    score = np.array([-vv[i][_] for _ in xrange(N)])
    label = np.array([(1 if _ == a else 0) for _ in xrange(N)])
    ap.append(average_precision_score(label, score))
    print 'ap', i, ap[-1]
    cnt_map = [[0 for j in xrange(6)] for i in xrange(args.sample_size)]
    for i in xrange(N):
        for j in xrange(N):
            cnt_map[fts[i][j][1]][fts[i][j][0]] += 1

