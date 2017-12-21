import numpy as np
import random


def get_data(t):
    try:
        batch = t.next()
    except StopIteration:
        t.reset()
        batch = t.next()
    return batch.data[0]


class BatchProvider:
    def __init__(self, F, is_train, size, sample_ratio=0.5, need_feat=False, start=None, end=None, is_valid=False, agent=None):
        self.F = F
        self.is_train = is_train
        self.size = size
        self.N = len(F)
        self.sample_ratio = sample_ratio
        self.need_feat = need_feat
        self.valid = is_valid
        self.cnt = 0
        self.agent = agent
        if start is not None:
            self.start, self.end = start, end
        else:
            self.start, self.end = 0, self.N
        self.reset()

    def reset(self):
        if self.valid:
            self.cnt = 0
        self.terminal = [True for _ in xrange(self.size)]
        self.A = -1
        self.B = -1
        self.cA = -1
        self.cB = -1
        self.tA = -1
        self.tB = -1
        self.curA = np.zeros((self.size, 3, 224, 112))
        self.curB = np.zeros((self.size, 3, 224, 112))

    def get_img(self, F, aug=False):
        return get_data(F)[0].asnumpy()

    def provide(self, preload=None):
        if random.random() < self.sample_ratio:
            a = b = (np.random.choice(self.end-self.start, 1) + self.start)[0]
            while len(self.F[a]) < 2:
                a = b = np.random.choice(self.N, 1)[0]
        else:
            a, b = (np.random.choice(self.end-self.start, 2, replace=False)+self.start)
        self.A, self.B = a, b
        if not a == b:
            self.cA, self.cB = np.random.choice(len(self.F[a]), 1)[0], np.random.choice(len(self.F[b]), 1)[0]
        else:
            self.cA, self.cB = np.random.choice(len(self.F[a]), 2, replace=False)
        self.tA, self.tB = np.random.choice(len(self.F[a][self.cA]), 1)[0], np.random.choice(len(self.F[b][self.cB]), 1)[0]
        print self.A, self.cA, self.tA
        for i in xrange(self.size):
            self.curA[i] = self.get_img(self.F[self.A][self.cA][self.tA], True)#self.A%self.N==self.B%self.N)
            self.curB[i] = self.get_img(self.F[self.B][self.cB][self.tB], True)#self.A%self.N==self.B%self.N)

        cur = [np.array(self.curA), np.array(self.curB)]
        return cur, self.A, self.B