import sys
sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx
import logging
import numpy as np
import argparse
import time
import random
import loss_layers
import lsoftmax
import loss_drop_layers
import pairwiseDropout
import scipy.io as sio
import h5py

# extract features for testing set 

def get_imRecordIter(name, input_shape, batch_size, kv, shuffle=False, aug=False):
    dataiter = mx.io.ImageRecordIter(
        path_imglist="%s/%s.lst" % (args.data_dir, name),
        path_imgrec="%s/%s.rec" % (args.data_dir, name),
        #mean_img="models/market_mean.bin",
        mean_r=128.0,
        mean_g=128.0,
        mean_b=128.0,
        rand_crop=aug,
        rand_mirror=aug,
        prefetch_buffer=4,
        preprocess_threads=3,
        shuffle=shuffle,
        label_width=1,
        data_shape=input_shape,
        batch_size=batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    return dataiter


def extract_feature(model, iterator, sav_name, num, batch_size):
    feature = np.zeros((num, args.feature_size))
    now = 0
    iterator.reset()
    for batch in iterator:
        data = batch.data[0]
        output = model.predict(data)
        real_size = batch_size - batch.pad
        output = output[:real_size]

        feature[now:now+real_size] = output
        now += real_size

    print feature.shape, now
    h5f = h5py.File(sav_name, 'w')
    h5f.create_dataset('feat', data=feature)
    h5f.close()
    #data = {'feat': feature}
    #sio.savemat(sav_name, data, do_compression=True)
    #np.savetxt(sav_name[:-4]+'.csv', feature)
#    with open(sav_name, "w") as f:
#        cPickle.dump(feature, f, protocol=cPickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(
        description='single domain car recog training')
    parser.add_argument('--gpus', type=str, default='6',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str,
                        default="/data3/matt/iLIDS-VID/recs",
                        help='data directory')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='the batch size')
    parser.add_argument('--feature-size', type=int, default=1024,
                        help='the feature size')
    parser.add_argument('--mode', type=str, default='ilds_baseline_b4',
                        help='model mode')
    parser.add_argument('--dataset', type=str, default='image_test',
                        help='dataset (test/query)')
    parser.add_argument('--kv-store', type=str,
                        default='device', help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=1,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default="test",
                        help='load model prefix')
    return parser.parse_args()


def load_checkpoint(prefix, epoch):
    # ssymbol = sym.load('%s-symbol.json' % prefix)
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (arg_params, aux_params)


args = parse_args()

print args
batch_size = args.batch_size
devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]

symbol, arg_params, aux_params = mx.model.load_checkpoint(
    'models/%s' % args.mode, args.model_load_epoch)

internals = symbol.get_internals()
symbol = internals["flatten_output"]
l2 = mx.symbol.L2Normalization(data=symbol, name='l2_norm')
kv = mx.kvstore.create(args.kv_store)
dataiter = get_imRecordIter(
    '%s' % args.dataset, (3, 224, 112), batch_size,
    kv, shuffle=False, aug=False)

model = mx.model.FeedForward(
    symbol=l2, ctx=devices, arg_params=arg_params,
    aux_params=aux_params, allow_extra_params=True)

num = len(file('%s/%s.lst' % (args.data_dir, args.dataset)).read().splitlines())
extract_feature(model, dataiter, 'features/%s-%s.mat' % (args.dataset, args.mode), num, batch_size)
print ('done')
