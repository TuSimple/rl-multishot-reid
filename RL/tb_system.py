class TensorBoardSystem:
    def __init__(self, pre, writer):
        self.tb_pool = {}
        self.pre = pre
        self.init_board()
        self.heartbeat = 0
        self.writer = writer
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def init_board(self):
        #pool_names = ['softmax_acc', 'triplet_loss', 'triplet_neg', 'triplet_pos', 'triplet_diff', 'triplet_ratio']
        Q_pool_names = ['neg_neg', 'pos_round', 'neg_ratio', 'neg_round', 'pos_pos', 'pos_ratio', 'pos_acc', 'neg_acc', 'Qvalue_0', 'Qvalue_1', 'Qvalue_2', 'Qvalue_3', 'Qgt_0', 'Qgt_1', 'Qdiff_2', 'Qdiff_3']#, 'epsilon']
        #for i in pool_names:
        #    add_board(tb_pool, i)
        '''for i in xrange(seq_len):
            pre = 'Q' + str(i) + '_'
            for j in Q_pool_names:
                add_board(tb_pool, pre + j)'''
        for j in Q_pool_names:
            self.add_board(self.pre + '_' + j)

    def add_board(self, name):
        self.tb_pool[name] = [0, 0]

    def update_board(self, name, v):
        self.tb_pool[name][0] += v
        self.tb_pool[name][1] += 1.0

    def get_board(self, name):
        if self.tb_pool[name][1] > 0.5:
            return (self.tb_pool[name][0] / self.tb_pool[name][1], self.heartbeat)
        else:
            return (0, 0)

    def put_board(self, Qvalue, action, t, delta, epsilon, rounds, dummy=False):
        '''for i in xrange(len(label)):
            update_board(tb_pool, pool_names[0], softmax_output[i] == label[i])
        for i in xrange(len(triplet_output)):
            t = triplet_output[i].asnumpy()
            for j in xrange(args.batch_size):
                update_board(tb_pool, pool_names[1 + i], t[j])'''
        act = action
        pre = self.pre
        self.update_board(('%s_Qvalue_%d' % (pre, act)), Qvalue[act])
        self.update_board(('%s_Qgt_%d' % (pre, t)), Qvalue[1 if t else 0])
        if act == 2:
            if t == 1:
                self.update_board('%s_neg_neg' % (pre), 0.0)
            else:
                self.update_board('%s_neg_neg' % (pre), 1.0)
            self.update_board('%s_neg_ratio' % (pre), 1.0)
        elif act == 3:
            if t == 1:
                self.update_board('%s_pos_pos' % (pre), 1.0)
            else:
                self.update_board('%s_pos_pos' % (pre), 0.0)
            self.update_board('%s_pos_ratio' % (pre), 1.0)
        else:
            if act == 1:
                if t == 1:
                    self.tp += 1.0
                else:
                    self.fp += 1.0
            else:
                if t == 1:
                    self.fn += 1.0
                else:
                    self.tn += 1.0
            self.update_board('%s_pos_ratio' % (pre), 0.0)
            self.update_board('%s_neg_ratio' % (pre), 0.0)
            self.update_board('%s_pos_pos' % (pre), t == act)
            if t == 1:
                self.update_board('%s_pos_acc' % (pre), act == 1)
                if not dummy:
                    self.update_board('%s_pos_round' % (pre), rounds)
            else:
                self.update_board('%s_neg_acc' % (pre), act == 0)
                if not dummy:
                    self.update_board('%s_neg_round' % (pre), rounds)
        #self.update_board('Q_epsilon', epsilon)

    def print_board(self):
        for i in self.tb_pool:
            v = self.get_board(i)
            if v[1] > 0:
                self.writer.add_scalar(i, v[0], v[1])
        if (self.tp + self.fp > 0) and (self.tp + self.fn > 0):
            precision = 1.0 * self.tp / (self.tp + self.fp)
            recall = 1.0 * self.tp / (self.tp + self.fn)
            gm = (precision * recall) ** 0.5
            acc = 1.0 * (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
            self.writer.add_scalar(self.pre + '_' + 'precision', precision, self.heartbeat)
            self.writer.add_scalar(self.pre + '_' + 'recall', recall, self.heartbeat)
            self.writer.add_scalar(self.pre + '_' + 'gm', gm, self.heartbeat)
            self.writer.add_scalar(self.pre + '_' + 'acc', acc, self.heartbeat)
            if precision + recall > 0:
                f1 = 2.0 * (precision * recall) / (precision + recall)
                self.writer.add_scalar(self.pre + '_' + 'f1', f1, self.heartbeat)
        self.heartbeat += 1
