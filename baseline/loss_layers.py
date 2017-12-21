import mxnet as mx
import numpy as np


class VerfiLoss(mx.operator.CustomOp):
    '''
    Verfication Loss Layer
    '''
    def __init__(self, grad_scale, threshd):
        self.grad_scale = grad_scale
        self.threshd = threshd
        self.eps = 1e-5

    def forward(self, is_train, req, in_data, out_data, aux):
        # print "forward"
        x = in_data[0]
        label = in_data[1].asnumpy()
        #print "verifi label", label
        n = x.shape[0]
        ctx = x.context
        # y = out_data[0]
        # y[:] = 0
        # print y.shape
        y = np.zeros((x.shape[0], ))
        #y = mx.nd.array((n, ), ctx=ctx)
        for i in range(x.shape[0]):
            #print "forward", i
            mask = np.zeros((n, ))
            if i<(x.shape[0]/2):
                pid = i + 1 if i % 2 == 0 else i - 1
                mask[i] = 1
                mask[pid] = 1                
            #mask[np.where(label == label[i])] = 1
            #print mask
            pos = np.sum(mask)
            mask = mx.nd.array(mask, ctx=ctx)
            diff = x[i] - x
            d = mx.nd.sqrt(mx.nd.sum(diff * diff, axis=1))
            d1 = mx.nd.maximum(0, self.threshd - d)
            z = mx.nd.sum(mask * d * d) / (pos + self.eps) \
                + mx.nd.sum((1 - mask) * d1 * d1) / (n - pos + self.eps)
            y[i] = z.asnumpy()[0]

        # y /= x.shape[0]
        self.assign(out_data[0], req[0], mx.nd.array(y, ctx=ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # print "backward"
        x = in_data[0]
        #label = in_data[1].asnumpy()
        n = x.shape[0]
        ctx = x.context
        grad = in_grad[0]
        grad[:] = 0
        for i in range(x.shape[0]):
            mask = np.zeros((1, n))
            #mask[np.where(label == label[i])] = 1
            if i<(x.shape[0]/2):
                    pid = i + 1 if i % 2 == 0 else i - 1
                    mask[0,i] = 1
                    mask[0,pid] = 1     
            pos = np.sum(mask)
            mask = mx.nd.array(mask, ctx=ctx)
            diff = x[i] - x
            d = mx.nd.sqrt(mx.nd.sum(diff * diff, axis=1))
            g1 = mx.nd.minimum(0, (d - self.threshd) / (d + self.eps))
            z = mx.nd.dot((1 - mask) * g1.reshape([1, n]), diff)[0]
            # print grad[i].shape, z.shape
            # grad[i] = z
            # print "z"
            grad[i] = mx.nd.dot(mask, diff)[0] / (pos + self.eps)\
                + mx.nd.dot((1 - mask) * g1.reshape([1, n]), diff)[0] / (n - pos + self.eps)             

        grad *= self.grad_scale



@mx.operator.register("verifiLoss")
class VerifiLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(VerifiLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.threshd = float(threshd)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0], )
        return [data_shape, label_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return VerfiLoss(self.grad_scale, self.threshd)


class TripletLoss(mx.operator.CustomOp):
    '''
    Triplet loss layer
    '''
    def __init__(self, grad_scale=1.0, threshd=0.5):
        self.grad_scale = grad_scale
        self.threshd = threshd

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = np.zeros((x.shape[0], ))
        ctx = x.context
        for i in range(x.shape[0] / 2):
            pid = i + 1 if i % 2 == 0 else i - 1
            nid = i + int(x.shape[0] / 2)
            pdiff = x[i] - x[pid]
            ndiff = x[i] - x[nid]
            y[i] = mx.nd.sum(pdiff * pdiff).asnumpy()[0] -\
                mx.nd.sum(ndiff * ndiff).asnumpy()[0] + self.threshd
            if y[i] < 0:
                y[i] = 0
        # y /= x.shape[0]
        self.assign(out_data[0], req[0], mx.nd.array(y, ctx=ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        y = out_data[0]
        grad = in_grad[0]
        grad[:] = 0
        for i in range(x.shape[0] / 2):
            pid = i + 1 if i % 2 == 0 else i - 1
            nid = i + int(x.shape[0] / 2)

            if y[i] > 0:
                grad[i] += x[nid] - x[pid]
                grad[pid] += x[pid] - x[i]
                grad[nid] += x[i] - x[nid]

        grad *= self.grad_scale 

        
        
@mx.operator.register("tripletLoss")
class TripletLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(TripletLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.threshd = float(threshd)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        # label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0], )
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return TripletLoss(self.grad_scale, self.threshd)
 
    
class CenterLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, num_class, alpha, scale=1.0):
        if not len(shapes[0]) ==2:
            raise ValuerError('dim for input_data should be 2 for CenterLoss')
        
        self.alpha = alpha
        self.batch_size = shapes[0][0]
        self.num_class = num_class
        self.scale = scale
        
    def forward(self, is_train, req, in_data, out_data, aux):
        x=in_data[0]
        labels = in_data[1].asnumpy()
        #print "center label", labels 
        diff = aux [0]
        center = aux[1]
        #loss=np.zeros((self.batch_size,1))
        
        for i in range(self.batch_size):
            diff[i] = in_data[0][i] - center[int(labels[i])]

        loss = mx.nd.sum(mx.nd.square(diff),axis=1) / self.batch_size /2
        self.assign(out_data[0], req[0], loss)       
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        diff = aux[0]
        center = aux[1]
        sum_  = aux[2]
        
        grad_scale = float(self.scale/self.batch_size)
        self.assign(in_grad[0], req[0], diff * grad_scale)
        
        
        #update the center
        labels = in_data[1].asnumpy()
        label_occur = dict()
        for i, label in enumerate(labels):
            label_occur.setdefault(int(label),  []).append(i)
            
        for label, sample_index in label_occur.items():
            sum_[:] = 0
            for i in sample_index:
                sum_ = sum_ + diff[i]
            delta_c = sum_ /(1+len(sample_index))
            center[label] += self.alpha * delta_c
            
@mx.operator.register("centerLoss")
class CenterLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_class, alpha, scale=1.0, batchsize=32):
        super(CenterLossProp, self).__init__(need_top_grad=False)
        
        self.num_class = int(num_class)
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.batchsize = int(batchsize)
        
    def list_arguments(self):
        return ['data',  'label']
    
    def list_outputs(self):
        return ['output']
    
    def list_auxiliary_states(self):
        return ['diff_bias', 'center_bias', 'sum_bias']
    
    def infer_shape(self, in_shape):  
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        
        #store diff, same shape as input batch
        diff_shape = [self.batchsize, data_shape[1]]

        #store the center of each clss, should be (num_class, d)
        center_shape = [self.num_class, diff_shape[1]]
        
        #computation buf
        sum_shape = [diff_shape[1], ]
        
        output_shape = (in_shape[0][0], )
        
        return [data_shape, label_shape], [output_shape], [diff_shape, center_shape, sum_shape]
    
    def create_operator(self, ctx, shapes, dtypes):
        return CenterLoss(ctx, shapes, dtypes, self.num_class, self.alpha, self.scale)
    
    
        
class lmnnLoss(mx.operator.CustomOp):
        '''
        LMNN Loss Layer = positive pairwise loss + triplet loss
        '''
        def __init__(self, epsilon, threshd):
            self.epsilon= epsilon      #epsilon is the trade-off parameter between positive pairwise and triplet loss(1: epsilon)
            self.threshd = threshd
            #self.pnr = pnr
    
        def forward(self, is_train, req, in_data, out_data, aux):
            # print "forward"
            x = in_data[0]
            #label=in_data[1].asnumpy()
            ctx = x.context
            y = mx.nd.zeros((x.shape[0], ), ctx=ctx)
            halfsize = x.shape[0]/2
            for i in range(halfsize):
                pid = i + 1 if i % 2 == 0 else i - 1
                pdiff = x[i] - x[pid]
                pdist = 0.5*mx.nd.sum(pdiff * pdiff)
                mask = np.ones((x.shape[0],))    #index mask for negative examples
                mask[i] = 0
                mask[pid] = 0
                mask = mx.nd.array(mask, ctx=ctx)
                ndiff = x[i] - x
                ndist = 0.5*mx.nd.sum(ndiff*ndiff,axis=1)
                distdiff = (pdist - ndist +self.threshd)*mask
                distdiff = mx.nd.sum(mx.nd.maximum(0, distdiff))/mx.nd.sum(mask)  
                y[i] = pdist+self.epsilon*distdiff

            self.assign(out_data[0], req[0], y)                     
            
    
        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            # print "backward"
            x = in_data[0]
            #label = in_data[1].asnumpy()
            ctx = x.context
            grad = in_grad[0]
            grad[:] = 0
            batchsize = x.shape[0]
            #label = in_data[1]
            #xhalf=x[halfsize:x.shape[0]]
            
            for i in range(batchsize/2):
                #print "gradient computation", i
                pid = i + 1 if i % 2 == 0 else i - 1
                grad[i] +=  x[i] - x[pid]
                grad[pid] += x[pid] - x[i]
                
                #pnr_index = np.random.binomial(n=1, p=self.pnr/batchsize, size=batchsize)
                #print pnr_index
                mask = np.ones((batchsize,))   #index mask for negative examples
                mask[i] = 0
                mask[pid] = 0   
                #mask=mask * pnr_index
                #print mask

                pdiff = x[i] - x[pid]
                pdist = 0.5 * mx.nd.sum(pdiff * pdiff)                
                ndiff = x[i] - x
                ndist = 0.5 * mx.nd.sum(ndiff * ndiff,axis=1)
                distdiff = pdist - ndist + self.threshd
                
                index = np.zeros((batchsize, ))
                index[np.where(distdiff.asnumpy()>0)]=1
                index=index * mask
                index=mx.nd.array(index,ctx=ctx)
                #print index
                
                ratio = distdiff * index / (mx.nd.sum(distdiff * index)+1e-5)
                ratio = mx.nd.Reshape(ratio, shape=(batchsize,1))
                #print ratio.asnumpy()
                ratio = mx.nd.broadcast_axis(ratio, axis=1, size=x.shape[1])
                #print ratio.asnumpy()
             
                grad[i] += mx.nd.sum((x-x[pid]) * ratio, axis=0) * self.epsilon
                grad[pid] += (x[pid]-x[i]) * self.epsilon * (mx.nd.sum(distdiff * index)/(mx.nd.sum(distdiff * index)+1e-5))
                grad += (x[i]-x) * ratio * self.epsilon
                
            self.assign(in_grad[0], req[0], grad)
            
@mx.operator.register("lmnnLoss")
class lmnnLossProp(mx.operator.CustomOpProp):
    def __init__(self, epsilon=1.0, threshd=0.5):
        super(lmnnLossProp, self).__init__(need_top_grad=False)
        self.epsilon = float(epsilon)
        self.threshd = float(threshd)
        #self.pnr = float(pnr)     #positive examples:negetive examples=1:pnr
    
    def list_arguments(self):
        return ['data']  # 'label']
    
    def list_outputs(self):
        return ['output']   

    def infer_shape(self, in_shape):
        data_shape = in_shape[0] 
        #label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0], )
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return lmnnLoss(self.epsilon, self.threshd)



                      
            
  
    
    
    
    
    
        
        
