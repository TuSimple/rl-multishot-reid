import numpy as np
import random
from segment_tree import SumSegmentTree, MinSegmentTree
from utils import copyto

class ReplayMemory:
    def __init__(self, replay_size, alpha=0.6):
        self.replay_size = replay_size
        self.cnt = 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < replay_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._storage = []
        self._maxsize = replay_size
        self._next_idx = 0

    def add(self, data):
        #new_data = []
        #for i in data:
        #    i.wait_to_read()
        #    new_data.append(copyto(i))
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        idx = self._next_idx
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha


    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        #print self._it_min.min(), weights
        weights = np.array(weights)
        weights /= np.sum(weights)
        ret = []
        for i in xrange(batch_size):
            ret.append(self._storage[idxes[i]])
        return (ret, idxes, weights)

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


