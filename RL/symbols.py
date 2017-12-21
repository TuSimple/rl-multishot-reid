import find_mxnet
import mxnet as mx
import importlib

from rnn_models import get_gru_cell, get_lstm_cell, lstm, gru

BN_EPS = 1e-5+1e-10


def sym_base_net(network, fix_gamma=False, is_train=False, global_stats=False, is_test=False):
    '''
    network structure
    '''
    if is_test:
        data = mx.symbol.Variable(name="data")
    else:
        data1 = mx.symbol.Variable(name="data1")
        data2 = mx.symbol.Variable(name="data2")
        data = mx.sym.Concat(*[data1, data2], dim=0, name='data')
    symbol = importlib.import_module('symbol_' + network).get_symbol(data, fix_gamma=fix_gamma, global_stats=global_stats)
    pooling = mx.symbol.Pooling(
        data=symbol, kernel=(1, 1), global_pool=True,
        pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')
    if is_test:
        l2 = mx.sym.L2Normalization(flatten)
        return l2
    else:
        split_flatten = mx.sym.SliceChannel(flatten, num_outputs=2, axis=0)
        return split_flatten[0], split_flatten[1]
    return None


def fusion_layer(data, num_hidden, num_layers, name, l2=False, weights=[], bias=[]):
    org_data = data
    for i in xrange(num_layers):
        data = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, name='%s%d'%(name,i), weight=weights[i], bias=bias[i])
        if i == 0:
            first_layer = data
        elif i == num_layers - 1:
            continue
        data = mx.sym.Activation(data=data, act_type='relu', name='%srelu%d'%(name,i))
    if l2:
        return mx.sym.Concat(*[mx.sym.L2Normalization(first_layer), mx.sym.L2Normalization(data)], dim=1)
    return mx.sym.Concat(*[org_data, data], dim=1)


def get_dist_sym(a, b):
    diff = a - b
    return mx.sym.sum(diff*diff, axis=1, keepdims=1)


def sym_DQN(data1, data2, num_sim, num_hidden, min_states, min_imgs, num_acts=4, fusion=False, is_train=False, bn=False, l2_norm=False, global_stats=False, no_his=True, debug=False, maxout=False, cls=False):
    #data1 = mx.sym.Dropout(data1)
    #data2 = mx.sym.Dropout(data2)

    featmaps = [mx.sym.SliceChannel(mx.sym.L2Normalization(data1), num_outputs=min_states, axis=0),
                mx.sym.SliceChannel(mx.sym.L2Normalization(data2), num_outputs=min_states, axis=0)]
    gs = featmaps
    ds, ts = [], []
    for i in xrange(min_states):
        d, t = [], []
        d.append(get_dist_sym(featmaps[0][i], featmaps[1][i]))
        t.append(mx.sym.sum(featmaps[0][i] * featmaps[1][i], axis=1, keepdims=1))
        for j in xrange(i):
            d.append(get_dist_sym(featmaps[0][i], featmaps[1][j]))
            d.append(get_dist_sym(featmaps[0][j], featmaps[1][i]))
            t.append(mx.sym.sum(featmaps[0][i] * featmaps[1][j], axis=1, keepdims=1))
            t.append(mx.sym.sum(featmaps[0][j] * featmaps[1][i], axis=1, keepdims=1))
        ds.append(d)
        ts.append(t)
        print i, len(d)


    featmap = mx.sym.abs(mx.sym.L2Normalization(data1) - mx.sym.L2Normalization(data2))
    featmaps = mx.sym.SliceChannel(featmap, num_outputs=min_states, axis=0, name='featmaps')
    W1, W2, W3 = mx.symbol.Variable(name="fc1_weight"), mx.symbol.Variable(name="fc2_weight"), mx.symbol.Variable(name="Qv_weight")
    b1, b2, b3 = mx.symbol.Variable(name="fc1_bias"), mx.symbol.Variable(name="fc2_bias"), mx.symbol.Variable(name="Qv_bias")
    if fusion:
        Wfus = [[mx.symbol.Variable(name="fus%d-%d_weight"%(i,j)) for j in xrange(2)] for i in xrange(3)]
        bfus = [[mx.symbol.Variable(name="fus%d-%d_bias"%(i,j)) for j in xrange(2)] for i in xrange(3)]
    ############# mini batch ###################
    ret, unsures, atts = [], [], []
    if True:
        for i in xrange(min_states):
            if i == 0:
                tmin, tmax, tsum = ts[0][0], ts[0][0], ts[0][0]
                dmin, dmax, dsum = ds[0][0], ds[0][0], ds[0][0]
            else:
                for j in xrange(len(ts[i])):
                    tmin = mx.sym.minimum(tmin, ts[i][j])
                    tmax = mx.sym.maximum(tmax, ts[i][j])
                    tsum = tsum + ts[i][j]
                for j in xrange(len(ds[i])):
                    dmin = mx.sym.minimum(dmin, ds[i][j])
                    dmax = mx.sym.maximum(dmax, ds[i][j])
                    dsum = dsum + ds[i][j]
            if i <= 1:
                agg = featmaps[0]
                feat = mx.sym.Concat(*[tmin, tmax, tsum / ((i + 1) * (i + 1)), dmin, dmax, dsum / ((i + 1) * (i + 1)),], dim=1)
            else:
                print i, len(unsures), len(ret)
                g1 = mx.sym.broadcast_mul(gs[0][0], unsures[0])
                g2 = mx.sym.broadcast_mul(gs[1][0], unsures[0])
                wsum = unsures[0]
                for j in xrange(1, i):
                    g1 = g1 + mx.sym.broadcast_mul(gs[0][j], unsures[j])
                    g2 = g2 + mx.sym.broadcast_mul(gs[1][j], unsures[j])
                    wsum = wsum + unsures[j]
                g1 = mx.sym.broadcast_div(g1, wsum)
                g2 = mx.sym.broadcast_div(g2, wsum)
                agg = mx.sym.abs(g1 - g2)
                feat = mx.sym.Concat(*[tmin, tmax, tsum / ((i + 1) * (i + 1)), dmin, dmax, dsum / ((i + 1) * (i + 1)),], dim=1)

            fm = featmaps[i]
            if fusion:
                print 'mini batch fusion on'
                agg = fusion_layer(agg, num_hidden, 2, 'fus_featmap1', weights=[Wfus[0][0], Wfus[0][1]], bias=[bfus[0][0], bfus[0][1]])#, l2=True)
                fm = fusion_layer(fm, num_hidden, 2, 'fus_featmap2', weights=[Wfus[1][0], Wfus[1][1]], bias=[bfus[1][0], bfus[1][1]])#, l2=True)
                feat = fusion_layer(feat, 32, 2, 'fus_feat', weights=[Wfus[2][0], Wfus[2][1]], bias=[bfus[2][0], bfus[2][1]])
            diff = mx.sym.Concat(*[agg, fm, feat], dim=1, name='diff%d'%i)
            #diff = mx.sym.Concat(*[agg, fm], dim=1, name='diff%d'%i)
            sir_fc1 = mx.sym.FullyConnected(data=diff, num_hidden=num_hidden,name='sirfc1-%d'%i, weight=W1, bias=b1)
            sir_relu1 = mx.sym.Activation(data=sir_fc1, act_type='relu', name='sirrl1-%d'%i)
            sir_fc2 = mx.sym.FullyConnected(data=sir_relu1, num_hidden=num_hidden,name='sirfc2-%d'%i, weight=W2, bias=b2)
            sir_relu2 = mx.sym.Activation(data=sir_fc2, act_type='relu', name='sirrl2-%d'%i)
            Q = mx.sym.FullyConnected(data=sir_relu2, num_hidden=num_acts,name='Qvalue-%d'%i, weight=W3, bias=b3)
            qsm = mx.sym.SoftmaxActivation(Q)
            qsm = mx.sym.BlockGrad(qsm)
            ret.append(Q)
            atts.append(qsm)
            if i + 1 < min_states:
                Q_sliced = mx.sym.SliceChannel(qsm, num_outputs=3, axis=1)
                unsures.append(1.0 - Q_sliced[2])

    Qvalue = mx.sym.Concat(*ret, dim=0, name='Qvalues')
    attss = mx.sym.Concat(*atts, dim=0, name='atts')
    return mx.sym.Group([Qvalue, attss])


if __name__ == '__main__':
    #sym = build_base_net('inception-bn')
    #mx.viz.print_summary(sym, {'data': (1, 3, 128, 64)})
    sym = sym_DQN(128, 128, num_acts=3, min_states=2, min_imgs=4, fusion=True)
    mx.viz.print_summary(sym, {'data1': (2, 1024), 'data2': (2, 1024)})
