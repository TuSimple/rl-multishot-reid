import numpy as np
import math
import argparse
import munkres
import hungarian
import random
import sys


cmcs = [1, 5, 10, 20]
cn = 4
single_sample_times, sample_rounds = 100, 10
match_points = 100

def norm_cnts(cnts, cnt):
    return [cnts[i] / cnt[0] for i in xrange(cn)]

def update_cnts(d, cnts, cnt, N, i):
    r = np.argsort(d)
    cur = -1
    for j in xrange(N):
        if r[j] == i:
            cur = j
            break
    for j in xrange(cn):
        if cur < cmcs[j]:
            cnts[j] += 1.0
    cnt[0] += 1.0
    print cur, norm_cnts(cnts, cnt)

def pooling_method(f, N):
    cam0 = []
    for i in xrange(N):
        p = np.zeros(1024)
        #p = np.full(1024, -1e100)
        for a in xrange(cnt_lst[i], cnt_lst[i + 1]):
            p += f[a]
            #p = np.maximum(p, f[a])
        p /= (cnt_lst[i + 1] - cnt_lst[i])
        cam0.append(p)
    cam1 = []
    for i in xrange(N):
        p = np.zeros(1024)
        #p = np.full(1024, -1e100)
        for a in xrange(cnt_lst[i + N], cnt_lst[i + N + 1]):
            p += f[a]
            #p = np.maximum(p, f[a])
        p /= (cnt_lst[i + N + 1] - cnt_lst[i + N])
        cam1.append(p)

    cam0, cam1 = np.array(cam0), np.array(cam1)
    for i in xrange(1024):
        norm = (cam0[:, i] * cam0[:, i]).sum() + (cam1[:, i] * cam1[:, i]).sum()
        norm = math.sqrt(norm)
        cam0[:, i] /= norm
        cam1[:, i] /= norm

    cnts, cnt = [0, 0, 0, 0], [0]
    for i in xrange(N):
        d = np.zeros(N)
        for j in xrange(N):
            t = (cam0[i] - cam1[j])
            d[j] += (t * t).sum()
        update_cnts(d, cnts, cnt, N, i)

    for i in xrange(N):
        d = np.zeros(N)
        for j in xrange(N):
            t = (cam1[i] - cam0[j])
            d[j] += (t * t).sum()
        update_cnts(d, cnts, cnt, N, i)

    print 'pooling method', norm_cnts(cnts, cnt)

def calc_mean(d):
    ret = np.zeros(len(d))
    for t in xrange(sample_rounds):
        for i in xrange(len(d)):
            x = 0.0
            for k in xrange(single_sample_times):
                a = random.randint(0, d[i].shape[0] - 1)
                b = random.randint(0, d[i].shape[1] - 1)
                x += d[i][a][b]
            x /= single_sample_times
            ret[i] += x
    return ret

def calc_median(d):
    ret = np.zeros(len(d))
    for t in xrange(sample_rounds):
        for i in xrange(len(d)):
            x = []
            for k in xrange(single_sample_times):
                a = random.randint(0, d[i].shape[0] - 1)
                b = random.randint(0, d[i].shape[1] - 1)
                x.append(d[i][a][b])
            x = sorted(x)
            ret[i] += x[single_sample_times / 2] + x[single_sample_times / 2 + 1]
    return ret

def calc_min(d):
    ret = np.zeros(len(d))
    for t in xrange(sample_rounds):
        for i in xrange(len(d)):
            x = 1e100
            for k in xrange(single_sample_times):
                a = random.randint(0, d[i].shape[0] - 1)
                b = random.randint(0, d[i].shape[1] - 1)
                x = min(x, d[i][a][b])
            ret[i] += x
    return ret

def calc_match(d):
    ret = np.zeros(len(d))
    for t in xrange(sample_rounds):
        for i in xrange(len(d)):
            choices_a = [random.randint(0, d[i].shape[0] - 1) for _ in xrange(match_points)]
            choices_b = [random.randint(0, d[i].shape[1] - 1) for _ in xrange(match_points)]
            mat = d[i][choices_a]
            mat = (mat.T)[choices_b]
            am = np.array(mat)
            match = hungarian.lap(am)[0]
            #M = munkres.Munkres()
            #match = M.compute(am)
            x = 0.0
            #g = []
            for p in xrange(len(match)):
            #for p in match:
                x += mat[i][match[i]]
                #g.append(mat[i][match[i]])
                #x += mat[p[0]][p[1]]
            #g.sort()
            #ret[i] += g[len(g) / 2] if len(g) % 2 == 1 else (g[len(g) / 2] + g[len(g) / 2 - 1]) * 0.5
            ret[i] += x
    return ret

def calc_order(d, rerank=False):
    ret = np.zeros(len(d))
    t = 10000
    for i in xrange(len(d)):
        t = min(d[i].shape[0], t)
        t = min(t, d[i].shape[1])
    for i in xrange(len(d)):
        if rerank:
            pass
        else:
            tp = min(d[i].shape[0], d[i].shape[1])
            choices_a = xrange(tp)
            choices_b = xrange(tp)
        mat = d[i][choices_a]
        mat = (mat.T)[choices_b]
        am = np.array(mat)
        M = munkres.Munkres()
        #print mat.shape
        match = M.compute(am)
        match = sorted(match)
        g = [mat[match[0][0]][match[0][1]]]
        for p in xrange(1, len(match)):
            g.append(mat[match[p][0]][match[p][1]])
            #for q in xrange(p - 1):
                #if match[p][1] > match[q][1]:
                #    ret[i] += 1
        g = sorted(g)
        for p in xrange(t):
            ret[i] += g[p]
        print len(match), ret[i]
    #print ret
    return ret

def other_method(f, N):
    cnts_median, cnt_median = [0, 0, 0, 0], [0]
    cnts_mean, cnt_mean = [0, 0, 0, 0], [0]
    cnts_min, cnt_min = [0, 0, 0, 0], [0]
    cnts_match, cnt_match = [0, 0, 0, 0], [0]
    cnts_order, cnt_order = [0, 0, 0, 0], [0]
    for i in xrange(N):
        d, na = [], cnt_lst[N + i + 1] - cnt_lst[N + i]
        for j in xrange(N):
            nb = cnt_lst[j + 1] - cnt_lst[j]
            t = np.zeros((nb, na))
            for b in xrange(cnt_lst[j], cnt_lst[j + 1]):
                for a in xrange(cnt_lst[N + i], cnt_lst[N + i + 1]):
                    g = f[a] - f[b]
                    t[b - cnt_lst[j], a - cnt_lst[N + i]] = (g * g).sum()
            d.append(t)
        print 'cam0', i
        update_cnts(calc_mean(d), cnts_mean, cnt_mean, N, i)
        update_cnts(calc_median(d), cnts_median, cnt_median, N, i)
        update_cnts(calc_min(d), cnts_min, cnt_min, N, i)
        update_cnts(calc_match(d), cnts_match, cnt_match, N, i)
        #update_cnts(calc_order(d), cnts_order, cnt_order, N, i)
        sys.stdout.flush()
    for i in xrange(N):
        d, na = [], cnt_lst[i + 1] - cnt_lst[i]
        for j in xrange(N):
            nb = cnt_lst[N + j + 1] - cnt_lst[N + j]
            t = np.zeros((nb, na))
            for b in xrange(cnt_lst[N + j], cnt_lst[N + j + 1]):
                for a in xrange(cnt_lst[i], cnt_lst[i + 1]):
                    g = f[a] - f[b]
                    t[b - cnt_lst[N + j], a - cnt_lst[i]] = (g * g).sum()
            d.append(t)
        print 'cam1', i
        update_cnts(calc_mean(d), cnts_mean, cnt_mean, N, i)
        update_cnts(calc_median(d), cnts_median, cnt_median, N, i)
        update_cnts(calc_min(d), cnts_min, cnt_min, N, i)
        update_cnts(calc_match(d), cnts_match, cnt_match, N, i)
        #update_cnts(calc_order(d), cnts_order, cnt_order, N, i)
        sys.stdout.flush()
    print 'min', norm_cnts(cnts_min, cnt_min)
    print 'mean', norm_cnts(cnts_mean, cnt_mean)
    print 'median', norm_cnts(cnts_median, cnt_median)
    print 'match', norm_cnts(cnts_match, cnt_match)
    sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Calc CMC Rank for ilds&prid dataset')
    parser.add_argument('--data', type=str,
                        default='features/image_test-prid_baseline_b4.csv',
                        help='data path')
    parser.add_argument('--list', type=str,
                        #default='/data3/matt/iLIDS-VID/recs/image_test.txt',
                        default='/data3/matt/prid_2011/recs/image_test.txt',
                        help='list path')
    return parser.parse_args()

args = parse_args()
print args

f, cnt_lst = np.loadtxt(args.data), np.loadtxt(args.list).astype(int)
N = cnt_lst.shape[0] / 2
for i in xrange(N):
    for a in xrange(cnt_lst[i] + 1, cnt_lst[i + 1]):
        f[a] += f[a - 1]
    for a in xrange(cnt_lst[i] + 1, cnt_lst[i + 1]):
        f[a] /= a - cnt_lst[i] + 1
for i in xrange(N):
    for a in xrange(cnt_lst[N + i] + 1, cnt_lst[N + i + 1]):
        f[a] += f[a - 1]
    for a in xrange(cnt_lst[N + i] + 1, cnt_lst[N + i + 1]):
        f[a] /= a - cnt_lst[N + i] + 1

pooling_method(f, N)
other_method(f, N)

