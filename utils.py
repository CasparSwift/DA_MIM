import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class LSTMTokenizer(object):
    def __init__(self, path):
        with open(os.path.join(path, 'word2id.txt'), 'r') as f:
            self.word2id = eval(f.read())

    def encode(self, words):
        return [self.word2id[w] for w in words]