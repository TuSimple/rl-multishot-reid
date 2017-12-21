import logging
import warnings
import find_mxnet
import mxnet as mx
import numpy as np
from mxnet.module import Module
from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet import ndarray as nd

COUNT_MAX = 1
USE_AVERAGE = False

class BaseModule(Module):
    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None):
        # count how many times gradients be added
        self.add_counter = 0
        self.count_max = COUNT_MAX
        super(BaseModule, self).__init__(symbol=symbol, data_names=data_names,
                                         label_names=label_names, logger=logger, context=context,
                                         fixed_param_names=fixed_param_names)

    def clear_gradients(self):
        """clear gradient
        """
        self.add_counter = 0
        for grads in self._exec_group.grad_arrays:
            for grad in grads:
                grad -= grad

    def aver_gradients(self, n):
        ''' get average gradients
        '''
        for grads in self._exec_group.grad_arrays:
            for grad in grads:
                grad /= float(n)

    def add_gradients_from_module(self, from_module):
        """add gradients
        """
        self.add_counter += 1
        gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                    from_module._exec_group.grad_arrays]
        for gradsto, gradsfrom in zip(self._exec_group.grad_arrays,
                                      gradfrom):
            for gradto, gradfrom in zip(gradsto, gradsfrom):
                gradto += gradfrom

        if self.add_counter == self.count_max:
            if USE_AVERAGE:
                self.aver_gradients(self.add_counter)
                self.update()
                self.clear_gradients()
            self.add_counter = 0

    def copy_from_module(self, from_module):
        """copy from another module
        """
        arg_params, aux_params = from_module.get_params()
        self.init_params(initializer=None, arg_params=arg_params,
                         aux_params=aux_params, force_init=True)

    def copy_param_from_module(self, from_module):
        arg_params, _ = from_module.get_params()
        _, aux_params = self.get_params()
        self.init_params(initializer=None, arg_params=arg_params,
                         aux_params=aux_params, force_init=True)

    def clip_gradients(self, threshold):
        """clip gradients
        """
        for grads in self._exec_group.grad_arrays:
            for grad in grads:
                grad -= grad - \
                    mx.nd.clip(grad, -1.0 * threshold, 1.0 * threshold).copy()


    def norm_clipping(self, threshold=1.0):
        """Clip the norm according to the threshold.
        All the gradients are concatenated to a single vector and the overall norm is calculated.
        Follows `[ICML2013] On the difficulty of training recurrent neural networks`
        Parameters
        ----------
        threshold : float, optional
        Returns
        -------
        norm_val : float
            The norm value. It could be used to measure whether the gradients are stable.
        """
        assert self.binded and self.params_initialized
        norm_val = self.get_global_norm_val()
        if norm_val > threshold:
            ratio = threshold / float(norm_val)
            for grads in self._exec_group.grad_arrays:
                for grad in grads:
                    grad[:] *= ratio
        return norm_val

    def get_global_norm_val(self):
        """Get the overall gradient norm ||W||_2
        Parameters
        ----------
        net : mx.mod.Module
        Returns
        -------
        norm_val : float
        """
        assert self.binded and self.params_initialized
        #TODO The code in the following will cause the estimated norm to be different for multiple gpus
        norm_val = 0.0
        for i in range(len(self._exec_group.grad_arrays[0])):
            norm_val += np.sqrt(
                sum([nd.norm(grads[i]).asnumpy()[0] ** 2
                     for grads in self._exec_group.grad_arrays]))
        norm_val /= float(len(self._exec_group.grad_arrays[0]))
        return norm_val
