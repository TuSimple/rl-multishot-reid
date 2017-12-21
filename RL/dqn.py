import sys
#sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx

from tensorboard import SummaryWriter
import logging
import numpy as np
import argparse
import random
import math
from sklearn.metrics import average_precision_score

from batch_provider import BatchProvider
from utils import get_imRecordIter
from replay_memory import ReplayMemory
from tb_system import TensorBoardSystem
from agent import Agent


def parse_args():
    parser = argparse.ArgumentParser(
        description='multishot recog training')
    parser.add_argument('--gpus', type=str, default='1',
                        help='the gpus will be used, e.g "0,1"')
    parser.add_argument('--data-dir', type=str,
                        default="/data3/matt/prid_2011",
                        help='data directory')
    parser.add_argument('--num-examples', type=int, default=10000,
                        help='the number of training examples')
    parser.add_argument('--num-train', type=int, default=20504,
                        help='the number of training examples')
    parser.add_argument('--num-valid', type=int, default=19529,
                        help='the number of training examples')
    parser.add_argument('--num-id', type=int, default=10,
                        help='the number of training ids')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='the batch size')
    parser.add_argument('--sample-size', type=int, default=4,
                        help='sample frames from each video')
    parser.add_argument('--patch-size', type=int, default=1,
                        help='size of single image patch from video')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='the initial learning rate')
    parser.add_argument('--num-epoches', type=int, default=100,
                        help='the number of training epochs')
    parser.add_argument('--mode', type=str, default='prid_video_match_%d-%d' % (4, 4),
                        help='save names of model and log')
    parser.add_argument('--verifi-threshd', type=float, default=0.9 + 2.3,
                        help='verification threshold')
    parser.add_argument('--kv-store', type=str,
                        default='device', help='the kvstore type')
    parser.add_argument('--network', type=str,
                        default='inception-bn', help='network name')
    parser.add_argument('--model-load-epoch', type=int, default=1,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default='prid_baseline_b4',
                        help='load model prefix')
    parser.add_argument('--q_duel', action='store_true', default=False,
                        help='if use duel network')
    parser.add_argument('--q_double', action='store_true', default=False,
                        help='if use double DQN')
    parser.add_argument('--q-weight', type=float, default=1.0,
                        help='DQN loss weight')
    parser.add_argument('--q-gamma', type=float, default=0.99,
                        help='DQN decay rate')
    parser.add_argument('--penalty', type=float, default=0.1,
                        help='DQN unsure penalty rate')
    parser.add_argument('--ob-epochs', type=int, default=1,
                        help='DQN observing epochs')
    parser.add_argument('--num_acts', type=int, default=3,
                        help='number of actions')
    parser.add_argument('--acts_per_round', type=int, default=3,
                        help='number of actions per round')
    parser.add_argument('--fix_gamma', action='store_true', default=False,
                        help='if fix_gamma in bn')
    parser.add_argument('--fix_penalty', action='store_true', default=False,
                        help='if fix penalty')
    parser.add_argument('--no_sim', action='store_true', default=False,
                        help='if no sim net')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='number of hidden neurons in Q learning fc layers')
    parser.add_argument('--target_freq', type=int, default=500,
                        help='number of hidden neurons in Q learning fc layers')
    parser.add_argument('--tisr', type=int, default=1,
                        help='time inverse lr step')
    parser.add_argument('--num_sim', type=int, default=128,
                        help='number of hidden neurons in similarity network')
    parser.add_argument('--lr_step', type=str, default='50,75',
                        help='number of epoches to shrink lr')
    parser.add_argument('--q_bn', action='store_true', default=False,
                        help='if add bn in qnet')
    parser.add_argument('--maxout', action='store_true', default=False,
                        help='if add maxout in qnet')
    parser.add_argument('--pr_alpha', type=float, default=0.6,
                        help='prioritized-replay alpha')
    parser.add_argument('--pr_beta', type=float, default=0.4,
                        help='prioritized-replay beta')
    parser.add_argument('--add_rewards', action='store_true', default=False,
                        help='if add rewards for single agent')
    parser.add_argument('--epsilon', action='store_true', default=False,
                        help='if epsilon learning')
    parser.add_argument('--pos_weight', type=float, default=1.0,
                        help='positive rewards weight')
    parser.add_argument('--e2e', action='store_true', default=False,
                        help='if e2e')
    parser.add_argument('--history', action='store_true', default=False,
                        help='if use history')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='choose the optimizer in {sgd, adam, rms}')
    parser.add_argument('--memory_size', type=int, default=1000,
                        help='memory buffer size')
    parser.add_argument('--final_epsilon', type=float, default=0.1,
                        help='final epsilon for exploration')
    parser.add_argument('--exp_ratio', type=float, default=0.1,
                        help='ratio for exploration in whole training process')
    parser.add_argument('--hinge', action='store_true', default=False,
                        help='if use hinge loss')
    parser.add_argument('--train-set', type=str, default='image_valid',
                        help='load model prefix')
    parser.add_argument('--valid-set', type=str, default='image_test',
                        help='load model prefix')
    parser.add_argument('--min-states', type=int, default=4,
                        help='minimum states for history')
    parser.add_argument('--min-imgs', type=int, default=0,
                        help='minimum imgs for each state')
    parser.add_argument('--precomputed', action='store_true', default=False,
                        help='if feature precomputed')
    parser.add_argument('--fusion', action='store_true', default=False,
                        help='if use data fusion')
    parser.add_argument('--total-forward', action='store_true', default=False,
                        help='if use data fusion')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='if print debug info')
    parser.add_argument('--crossvalid', action='store_true', default=False,
                        help='if do cross validation')
    return parser.parse_args()


args = parse_args()
logging.basicConfig(filename='log/%s.log' % args.mode, level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.info(args)
logdir = './tblog/' + args.mode
print args
batch_size = args.batch_size
num_epoch = args.num_epoches
devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]
lr = args.lr

agent = Agent(args, devices[0])

prefix = 'models/%s' % args.mode
if args.crossvalid:
    summary_writer = SummaryWriter(logdir)
    tbs_V = TensorBoardSystem('V', summary_writer)

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

iterations = args.num_examples
memory = ReplayMemory(replay_size=args.memory_size, alpha=args.pr_alpha)
epsilon = 1.0
final_epsilon = args.final_epsilon
rand_ep, fix_ep = 0, int(args.num_epoches * args.exp_ratio)
epsilon_shr = (epsilon - final_epsilon) / (fix_ep - rand_ep) / iterations

max_penalty = 1

frf = open(('figurelog/%s' % args.mode), 'w')

for e in xrange(args.num_epoches):
    if args.verbose:
        print 'Epoch', e
    for batch in xrange(iterations):
        if args.verbose:
            print 'Epoch', e, 'batch', batch
        cur, a, b, cur_id = train.provide()
        y = ((a %N) == (b % N))
        data_batch = agent.wash_data(cur)
        Qvalue = agent.get_Qvalue(data_batch, is_train=False)
        if args.verbose:
            print 'forward', Qvalue
            qs = agent.Q.get_outputs()[1].asnumpy()
            print qs
            print  qs.max(), qs.min(), qs.mean(), qs.std()
        Qvalue_softmax = mx.nd.SoftmaxActivation(mx.nd.array(Qvalue, ctx=devices[0]) / epsilon / 5).asnumpy()
        reward, action, i = [0 for _ in xrange(args.min_imgs)], [-1 for _ in xrange(args.min_imgs)], args.min_imgs
        while i < args.sample_size:
            if args.total_forward:
                if i + 1 < args.sample_size:
                    k = 2
                else:
                    Q_choice = np.argmax(Qvalue[i, :2]) if args.epsilon else np.random.choice(args.num_acts, 1, p=Qvalue_softmax[i, :2])[0]
                    if random.random() <= epsilon and args.epsilon:
                        k = random.randrange(2)
                    else:
                        k = Q_choice
            else:
                Q_choice = np.argmax(Qvalue[i]) if args.epsilon else np.random.choice(args.num_acts, 1, p=Qvalue_softmax[i])[0]
                if random.random() <= epsilon and args.epsilon:
                    k = random.randrange(args.num_acts)
                else:
                    k = Q_choice
            cls = k % args.acts_per_round
            step = k - 1
            if cls >= 2:
                if i + step >= args.sample_size:
                    r = -max_penalty
                    terminal = True
                else:
                    #r = -args.penalty * step #- step_penalty_rate / (args.sample_size - t[i])
                    r = -args.penalty * (2.0 - (0.5 ** (step - 1)))
                    #if not args.fix_penalty else 1.0 - 0.5 * cnt
                    terminal = False
            else:
                r = 1 if cls == y else -max_penalty #(-10 if y else -10)
                terminal = True
            if args.pos_weight > 1:
                if y:
                    r *= args.pos_weight
            else:
                if not y:
                    r /= args.pos_weight
            reward.append(r)
            action.append(k)
            if args.verbose:
                print i, (a, b), Qvalue[i], k, (y, cls), r
            if terminal:
                break
            i += step
        #memory.add(dict(cur = cur_id, reward=reward, action=action, y=y, cnt=1))
        memory.add(dict(cur = cur, reward=reward, action=action, y=y, cnt=1))
        if rand_ep <= e < fix_ep:
            epsilon -= epsilon_shr
            epsilon = max(epsilon, final_epsilon)
        if e * args.num_examples + batch < 50:#args.num_examples / 2:
            continue

        replays, idxes, weights = memory.sample(args.batch_size, args.pr_beta)
        new_weights = []
        for b in xrange(args.batch_size):
            cur, reward, action, y = replays[b]['cur'], replays[b]['reward'], replays[b]['action'], replays[b]['y']
            data_batch, delta_sum = agent.wash_data(cur), 0
            Qvalue = agent.get_Qvalue(data_batch, is_train=True)
            grad, r, grad_norm = np.zeros((args.sample_size, args.num_acts)), 0, 0
            t = args.min_imgs
            for i in xrange(len(action) - 1):
                t += action[i] - 1
            for i in xrange(len(action) - 1, -1, -1):
                if i < len(action) - 1:
                    r = reward[i] + args.q_gamma * max(last_Q)#min(1.0, max(last_Q))
                else:
                    r = reward[i]
                #last_Q_, last_Q = Qvalue_[t], Qvalue[t]
                last_Q = Qvalue[t]
                if args.verbose:
                    print i, t, action[i], y, Qvalue[t], r,
                delta = -r + Qvalue[t, action[i]]
                if not args.total_forward:
                    delta /= len(action)
                if abs(delta) > 1:
                    delta /= abs(delta)
                if args.hinge:
                    if (y and action == 1 or not y and action == 0) and delta > 0:
                        clipped_delta = 0
                    elif (y and action == 0 or not y and action == 1) and delta < 0:
                        clipped_delta = 0
                    else:
                        clipped_delta = delta
                else:
                    clipped_delta = delta
                #if abs(clipped_delta) > 1:
                #    clipped_delta /= abs(clipped_delta)
                grad[t, action[i]] = clipped_delta
                grad_norm += (clipped_delta) * (clipped_delta)
                if args.verbose:
                    print delta, grad[i]
                delta_sum += abs(delta)
                if i > 0:
                    t -= (action[i - 1] - 1)
                if args.total_forward:
                    break
            new_weights.append(1)
            replays[b]['cnt'] += 1
            replays[b]['delta'] = delta
            grad_norm = math.sqrt(grad_norm)
            if args.verbose:
                print 'grad norm =', grad_norm
            agent.update([mx.nd.array(grad, ctx=devices[0]), mx.nd.zeros(agent.Q.get_outputs()[1].shape, ctx=devices[0])])
        memory.update_priorities(idxes, new_weights)
        if args.verbose:
            print 'gamma =', args.q_gamma, 'epsilon =', epsilon
    #args.q_gamma = min(0.99, 0.99 * args.q_gamma + 0.01)

    if (e+1) % 1 == 0:
        agent.save(e+1)

    if not args.crossvalid:
        continue

    valid.reset()
    batch, valid_cnt, vv, vpool = 0, 0, np.zeros((N*2, N)), set()
    vs, vt = [0 for i in xrange(N+N)], [0 for i in xrange(N+N)]
    fts = [[0 for _2 in xrange(N)] for _1 in xrange(N)]
    while valid_cnt < N*N:
        if args.verbose:
            print 'Epoch', e, 'valid', batch, 'vc', valid_cnt
        batch += 1
        cur, a, b = valid.provide()
        y = ((a %N) == (b % N))
        data_batch = agent.wash_data(cur)
        Qvalue = agent.get_Qvalue(data_batch, is_train=False)
        if args.verbose:
            print Qvalue
            print agent.Q.get_outputs()[1].asnumpy().max(), agent.Q.get_outputs()[1].asnumpy().min()
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
                r = 1 if cls == y else -max_penalty #(-10 if y else -10)
                terminal = True
            if args.pos_weight > 1:
                if y:
                    r *= args.pos_weight
            else:
                if not y:
                    r /= args.pos_weight
            if args.verbose:
                print 'valid', i, (a, b), Qvalue[i], k, (y, cls), r
            va, vb = a, b % N
            if (va, vb) not in vpool:
                vs[va] += r
                vs[vb+N] += r
            if terminal:
                if (va, vb) not in vpool:#vv[va][vb] == 0:
                    fts[va][vb] = (k + (3 if va == vb else 0), i)
                    vpool.add((va, vb))
                    valid_cnt += 1
                    vv[va][vb] = Qvalue[i][0] - Qvalue[i][1]
                    vt[va] += i + 1
                    vv[vb+N][va] += vv[va][vb]
                    vt[vb+N] += i + 1
                    if args.verbose:
                        print va, vb, vv[va][vb], r
            tbsQvalue = np.zeros(3)
            tbsQvalue[min(2, cls)] = Qvalue[i, k]
            tbs_V.put_board(tbsQvalue, min(2, cls), y, r, epsilon, i + 1, dummy=False)
            if terminal:
                break
            i += step
        if valid_cnt % 100 == 0:
            tbs_V.print_board()
    for i in xrange(N*2):
        a, r = i % N, 0
        for b in xrange(N):
            if a != b and vv[i][b] <= vv[i][a]:
                r += 1
        for k in xrange(4):
            cmcs[k].append(1.0 if r < cmcn[k] else 0.0)
            if len(cmcs[k]) >= N*4:
                summary_writer.add_scalar(args.mode[0] + ('_CMC%d'%cmcn[k]), sum(cmcs[k][-N*2:]) / (N*2), len(cmcs[k]) - N*2)
        vscores += [vs[i]]
        vturns += [vt[i]]
        score = np.array([-vv[i][_] for _ in xrange(N)])
        label = np.array([(1 if _ == a else 0) for _ in xrange(N)])
        ap.append(average_precision_score(label, score))
        if args.verbose:
            print 'ap', i, ap[-1]
        if len(ap) >= N*4:
            summary_writer.add_scalar(args.mode[0] + '_MAP', sum(ap[-N*2:]) / (N*2), len(ap) - N*2)
        if len(ap) >= N*4:
            summary_writer.add_scalar(args.mode[0] + ('_scores'), sum(vscores[-N*2:]) / (N*2), len(vscores) - N*2)
            summary_writer.add_scalar(args.mode[0] + ('_turns'), sum(vturns[-N*2:]) / (N*2), len(vturns) - N*2)
    frf.write('%d %.3f %.3f %.3f %.3f %.3f %.3f\n'%(e, sum(cmcs[0][-N*2:]) / (N*2), sum(cmcs[1][-N*2:]) / (N*2), sum(cmcs[2][-N*2:]) / (N*2), sum(cmcs[3][-N*2:]) / (N*2), sum(ap[-N*2:]) / (N*2), sum(vturns[-N*2:])*1.0 / (N*N*2)))
    cnt_map = [[0 for j in xrange(6)] for i in xrange(args.sample_size)]
    for i in xrange(N):
        for j in xrange(N):
            cnt_map[fts[i][j][1]][fts[i][j][0]] += 1
    for i in xrange(args.sample_size):
        for j in xrange(6):
            frf.write(str(cnt_map[i][j]) + ' ')
        frf.write('\n')
    frf.flush()

frf.close()
