import find_mxnet
import mxnet as mx
from img_lib import ImgLibrary


def dist(a, b):
    diff = mx.nd.L2Normalization(mx.nd.expand_dims(a, axis=0)) - mx.nd.L2Normalization(mx.nd.expand_dims(b, axis=0))
    return mx.nd.sum(diff * diff).asnumpy()[0]


class TimeInvScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, step, stop_factor_lr=1e-8):
        super(TimeInvScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        self.step = step
        self.stop_factor_lr = stop_factor_lr

    def __call__(self, num_update):
        t = num_update / self.step
        lr = self.base_lr * 1.0 / (1.0 + t)
        if lr < self.stop_factor_lr:
            lr = self.stop_factor_lr
        return lr


def load_checkpoint(prefix, epoch):
    # symbol = sym.load('%s-symbol.json' % prefix)
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        #if name in ['triplet_match', 'triplet', 'lmnn', 'lsoftmax', 'lsoftmax_weight', 'lsoftmax_label']:
        #    continue
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (arg_params, aux_params)


def copyto(x):
    return x.copyto(x.context)


def get_imRecordIter(args, name, input_shape, batch_size, kv=None, shuffle=False, aug=False, even_iter=False):
    '''
    get iterator use ImgLibrary or ImageRecordIter
    '''
    if even_iter:
        aug_params = {}
        aug_params['resize'] = 128
        aug_params['rand_crop'] = aug
        aug_params['rand_mirror'] = aug
        aug_params['input_shape'] = input_shape
        aug_params['mean'] = 128.0

        dataiter = ImgLibrary(
            '%s/%s.lst' % (args.data_dir, name),
            batch_size=batch_size,
            aug_params=aug_params,
            shuffle=shuffle,
            data_dir = args.data_dir)
    else:
        if aug:
            dataiter = mx.io.ImageRecordIter(
                path_imglist="%s/%s.lst" % (args.data_dir, name),
                path_imgrec="%s/%s.rec" % (args.data_dir, name),
                # mean_img="models/market_mean.bin",
                mean_r=128.0,
                mean_g=128.0,
                mean_b=128.0,
                rand_crop=True,
                rand_mirror=True,
                #max_random_contrast=0.1,
                #max_random_illumination=0.1,
                #max_aspect_ratio=0.1,
                #max_shear_ratio=0.2,
                #random_h=10,
                #random_s=10,
                #random_l=10,
                #max_random_contrast=0.2,
                #max_random_illumination=0.2,
                #max_aspect_ratio=0.2,
                #max_shear_ratio=0.2,
                #random_h=30,
                #random_s=30,
                #random_l=30,
                prefetch_buffer=4,
                preprocess_threads=4,
                shuffle=shuffle,
                label_width=1,
                round_batch=True,
                data_shape=input_shape,
                batch_size=batch_size,)
                #num_parts=kv.num_workers,
                #part_index=kv.rank)
        else:
            dataiter = mx.io.ImageRecordIter(
                path_imglist="%s/%s.lst" % (args.data_dir, name),
                path_imgrec="%s/%s.rec" % (args.data_dir, name),
                # mean_img="models/market_mean.bin",
                mean_r=128.0,
                mean_g=128.0,
                mean_b=128.0,
                prefetch_buffer=4,
                preprocess_threads=4,
                shuffle=shuffle,
                label_width=1,
                round_batch=True,
                data_shape=input_shape,
                batch_size=batch_size,)
                #num_parts=kv.num_workers,
                #part_index=kv.rank)

    return dataiter
