import numpy as np
import random

class BatchProvider:
    def __init__(self, F, lst, is_train, size, sample_ratio=0.5, need_feat=False, start=None, end=None, is_valid=False, agent=None):
        self.F = F
        self.lst = lst
        self.is_train = is_train
        self.size = size
        self.N = lst.shape[0] / 2
        self.sample_ratio = sample_ratio
        self.need_feat = need_feat
        self.valid = is_valid
        self.cnt = 0
        self.agent = agent
        if start is not None:
            self.start, self.end = start, end
        else:
            self.start, self.end = 0, self.N
        self.vid = []
        for i in xrange(self.N + self.N):
            for j in xrange(lst[i + 1] - lst[i]):
                self.vid.append(i)
        self.epoch_rounds = lst[-1] * 2
        self.first_imgs = [(i, j) for i in xrange(lst[-1]) for j in xrange(2)]
        random.shuffle(self.first_imgs)
        self.vid = []
        for i in xrange(self.N + self.N):
            for j in xrange(lst[i + 1] - lst[i]):
                self.vid.append(i)
        self.reset()
        self.hit_cnt = np.zeros(self.epoch_rounds / 2)
        self.img_rank = []
        for i in xrange(self.N + self.N):
            g = []
            for j in xrange(lst[i], lst[i + 1]):
                g.append(j)
            random.shuffle(g)
            self.img_rank.append(g)


    def reset(self):
        if self.valid:
            self.cnt = 0
        self.terminal = [True for _ in xrange(self.size)]
        self.A = -1
        self.B = -1
        self.curA = np.zeros((self.size, 3, 224, 112))
        self.curB = np.zeros((self.size, 3, 224, 112))


    def get_img(self, i, aug=False):
        idx = random.randrange(self.lst[i], self.lst[i + 1])
        self.hit_cnt[idx] += 1
        return self.F.get_single(idx, aug), idx


    def provide(self, preload=None):
        if preload is None:
            if self.valid:
                next = self.cnt
                self.cnt += 1
                self.A = next / self.N
                self.B = next % self.N
                self.B += self.N
                for i in xrange(self.size):
                    self.curA[i] = self.F.get_single(self.img_rank[self.A][i % len(self.img_rank[self.A])])
                    self.curB[i] = self.F.get_single(self.img_rank[self.B][i % len(self.img_rank[self.B])])
            else:
                first_img = self.first_imgs[self.cnt % self.epoch_rounds]
                self.cnt += 1
                a = self.vid[first_img[0]]
                if a < self.N:
                    if first_img[1] == 1:
                        b = a + self.N
                    else:
                        b = self.vid[random.randrange(self.lst[self.N], self.lst[self.N+self.N])]
                        while b == a + self.N:
                            b = self.vid[random.randrange(self.lst[self.N], self.lst[self.N+self.N])]
                else:
                    if first_img[1] == 1:
                        b = a - self.N
                    else:
                        b = self.vid[random.randrange(self.lst[0], self.lst[self.N])]
                        while b == a - self.N:
                            b = self.vid[random.randrange(self.lst[0], self.lst[self.N])]
                self.A, self.B = a, b
                idx = []
                for i in xrange(self.size):
                    self.curA[i], ida = self.get_img(self.A, True)#self.A%self.N==self.B%self.N)
                    self.curB[i], idb = self.get_img(self.B, True)#self.A%self.N==self.B%self.N)
                    idx.append((ida, idb))
        else:
            for i in xrange(self.size):
                self.curA[i], self.curB[i] = self.F.get_single(preload[i][0]), self.F.get_single(preload[i][1])

        cur = [np.array(self.curA), np.array(self.curB)]
        if not self.valid:
            if preload is None:
                return cur, self.A, self.B, idx
            else:
                return cur
        return cur, self.A, self.B