import sys
sys.path.insert(0, "mxnet/python/")
import find_mxnet
import mxnet as mx
import logging
import numpy as np
import argparse
from mxnet.optimizer import SGD
import loss_layers
import lsoftmax 
from verifi_iterator import verifi_iterator
from even_iterator import Even_iterator
import importlib


def build_network(symbol, num_id, batchsize):
    '''
    network structure
    '''
    # concat = internals["ch_concat_5b_chconcat_output"]
    pooling = mx.symbol.Pooling(
        data=symbol, kernel=(1, 1), global_pool=True,
        pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')
    l2 = mx.symbol.L2Normalization(data=flatten, name='l2_norm')    
    dropout = l2#mx.symbol.Dropout(data=l2, name='dropout1')       
    
    if args.lsoftmax:
        #fc1 = mx.symbol.Custom(data=flatten, num_hidden=num_id, beta=1000, margin=3, scale=0.9999, beta_min=1, op_type='LSoftmax', name='lsoftmax')
        fc1 = mx.symbol.LSoftmax(data=flatten, num_hidden=num_id, beta=1000, margin=4, scale=0.99999, beta_min=3, name='lsoftmax')
    else:
        fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_id, name='cls_fc1')

    softmax1 = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
   
    outputs = [softmax1]
    if args.verifi: 
        verifi = mx.symbol.Custom(data=dropout, grad_scale=1.0, threshd=args.verifi_threshd, op_type='verifiLoss', name='verifi')
        outputs.append(verifi)   
    
    if args.triplet:
        triplet = mx.symbol.Custom(data=dropout, grad_scale=1.0, threshd=args.triplet_threshd, op_type='tripletLoss', name='triplet')
        outputs.append(triplet)    

    if args.lmnn:         
        lmnn = mx.symbol.Custom(data=dropout, epsilon=0.1, threshd=0.9, op_type='lmnnLoss', name='lmnn')     
        outputs.append(lmnn)   

    if args.center:
        center = mx.symbol.Custom(data=dropout, op_type='centerLoss', name='center', num_class=num_id, alpha=0.5, scale=1.0, batchsize=batchsize)        
        outputs.append(center)
        
    return mx.symbol.Group(outputs)


class Multi_Metric(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""
    def __init__(self, num=None, cls=1):
        super(Multi_Metric, self).__init__('multi-metric', num)
        self.cls = cls

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)
        # classification loss
        for i in range(self.cls):
            pred_label = mx.nd.argmax_channel(preds[i])
            pred_label = pred_label.asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')
        
            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

        # verification losses
        for i in range(self.cls, len(preds)):
            pred = preds[i].asnumpy()
            if self.num is None:
                self.sum_metric += np.sum(pred)
                self.num_inst += len(pred)
            else:
                self.sum_metric[i] += np.sum(pred)
                self.num_inst[i] += len(pred)

def get_imRecordIter(name, input_shape, batch_size, kv, shuffle=False, aug=False, even_iter=False):
    '''
    get iterator use even_iterator or ImageRecordIter
    '''
    if even_iter:
        aug_params = {}
        aug_params['resize'] = 128
        aug_params['rand_crop'] = aug
        aug_params['rand_mirror'] = aug
        aug_params['input_shape'] = input_shape
        aug_params['mean'] = 128.0

        dataiter = Even_iterator(
            '%s/%s.lst' % (args.data_dir, name),
            batch_size=batch_size / 2,
            aug_params=aug_params,
            shuffle=shuffle)
    else:
        dataiter = mx.io.ImageRecordIter(
            path_imglist="%s/%s.lst" % (args.data_dir, name),
            path_imgrec="%s/%s.rec" % (args.data_dir, name),
            # mean_img="models/market_mean.bin",
            mean_r=128.0,
            mean_g=128.0,
            mean_b=128.0,
            rand_crop=aug,
            rand_mirror=aug,
            prefetch_buffer=4,
            preprocess_threads=3,
            shuffle=shuffle,
            label_width=1,
            round_batch=False,
            data_shape=input_shape,
            batch_size=batch_size / 2,
            num_parts=kv.num_workers,
            part_index=kv.rank)

    return dataiter


def get_iterators(batch_size, input_shape, train, test, kv, gpus=1):
    '''
    use image lists to generate data iterators
    '''
    train_dataiter1 = get_imRecordIter(
        '%s_even' % train, input_shape, batch_size,
        kv, shuffle=args.even_iter, aug=True, even_iter=args.even_iter)
    train_dataiter2 = get_imRecordIter(
        '%s_rand' % train, input_shape, batch_size,
        kv, shuffle=True, aug=True)
    val_dataiter1 = get_imRecordIter(
        '%s_even' % test, input_shape, batch_size,
        kv, shuffle=False, aug=False, even_iter=args.even_iter)
    val_dataiter2 = get_imRecordIter(
        '%s_rand' % test, input_shape, batch_size,
        kv, shuffle=False, aug=False)

    return verifi_iterator(
        train_dataiter1, train_dataiter2, use_verifi=args.verifi, use_center=args.center, use_lsoftmax=args.lsoftmax, gpus=gpus), \
        verifi_iterator(
            val_dataiter1, val_dataiter2, use_verifi=args.verifi, use_center=args.center, use_lsoftmax=args.lsoftmax, gpus=gpus)


def parse_args():
    parser = argparse.ArgumentParser(
        description='single domain car recog training')
    parser.add_argument('--gpus', type=str, default='5',
                        help='the gpus will be used, e.g "0,1"')
    parser.add_argument('--data-dir', type=str,
                        default="/data3/matt/iLIDS-VID/recs",
                        help='data directory')
    parser.add_argument('--num-examples', type=int, default=20000,
                        help='the number of training examples')
    parser.add_argument('--num-id', type=int, default=150,
                        help='the number of training ids')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='the initial learning rate')
    parser.add_argument('--num-epoches', type=int, default=1,
                        help='the number of training epochs')
    parser.add_argument('--mode', type=str, default='ilds_baseline_b4',
                        help='save names of model and log')
    parser.add_argument('--lsoftmax', action='store_true', default=False,
                        help='if use large margin softmax')
    parser.add_argument('--verifi-label', action='store_true', default=False,
                        help='if add verifi label')
    parser.add_argument('--verifi', action='store_true', default=False,
                        help='if use verifi loss')
    parser.add_argument('--triplet', action='store_true', default=False,
                        help='if use triplet loss')
    parser.add_argument('--lmnn', action='store_true', default=True,
                        help='if use LMNN loss')    
    parser.add_argument('--center', action='store_true', default=False,
                        help='if use center loss')
    parser.add_argument('--verifi-threshd', type=float, default=0.9,
                        help='verification threshold')
    parser.add_argument('--triplet-threshd', type=float, default=0.9,
                        help='triplet threshold')
    parser.add_argument('--train-file', type=str, default="image_train",
                        help='train file')
    parser.add_argument('--test-file', type=str, default="image_valid",
                        help='test file')
    parser.add_argument('--kv-store', type=str,
                        default='device', help='the kvstore type')
    parser.add_argument('--network', type=str,
                        default='inception-bn', help='network name')
    parser.add_argument('--model-load-epoch', type=int, default=126,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default='inception-bn',
                        help='load model prefix')
    parser.add_argument('--even-iter', action='store_true', default=False,
                        help='if use even iterator')
    return parser.parse_args()


def load_checkpoint(prefix, epoch):
    # symbol = sym.load('%s-symbol.json' % prefix)
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
num_epoch = args.num_epoches
devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]
lr = args.lr
num_images = args.num_examples


arg_params, aux_params = load_checkpoint(
    'models/%s' % args.model_load_prefix, args.model_load_epoch)

symbol = importlib.import_module(
    'symbol_' + args.network).get_symbol()

#batchsize4center=batch_size / len(devices)
net = build_network(symbol, num_id=args.num_id, batchsize= batch_size)

kv = mx.kvstore.create(args.kv_store)
train, val = get_iterators(
    batch_size=batch_size, input_shape=(3, 224, 112),
    train=args.train_file, test=args.test_file, kv=kv, gpus=len(devices))
print train.batch_size
#train = get_imRecordIter(args.train_file, (3, 224, 112), batch_size, kv)
#val = get_imRecordIter(args.test_file, (3, 224, 112), batch_size, kv)

stepPerEpoch = int(num_images * 2 / batch_size)
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
    step=[stepPerEpoch * x for x in [50, 75]], factor=0.1)
init = mx.initializer.Xavier(
    rnd_type='gaussian', factor_type='in', magnitude=2)

arg_names = net.list_arguments()
sgd = SGD(learning_rate=args.lr, momentum=0.9,
          wd=0.0005, clip_gradient=10, lr_scheduler=lr_scheduler,
          rescale_grad=1.0 / batch_size)


logging.basicConfig(filename='log/%s.log' % args.mode, level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.info(args)

print ('begining of mx.model.feedforward')

model = mx.model.FeedForward(
    symbol=net, ctx=devices, num_epoch=num_epoch, arg_params=arg_params,
    aux_params=aux_params, initializer=init, optimizer=sgd)

prefix = 'models/%s' % args.mode
num = 1
if args.verifi:
    num += 1
if args.triplet:
    num += 1
if args.lmnn:
        num += 1
if args.center:
    num += 1
    

eval_metric=Multi_Metric(num=num, cls=1)
epoch_end_callback=mx.callback.do_checkpoint(prefix)
batch_end_callback=mx.callback.Speedometer(batch_size=batch_size)
print ('begining of model.fit')
model.fit(X=train, eval_data=val, eval_metric=eval_metric, logger=logger, epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback)
print('done')