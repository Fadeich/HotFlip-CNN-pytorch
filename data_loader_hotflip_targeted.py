#!/usr/bin/env python3
import csv
import random
import numpy as np
from termcolor import colored
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from data_loader_hotflip_beam import AGNEWs_HotFlip_Beam

class AGNEWs_HotFlip_Targeted(AGNEWs_HotFlip_Beam):
    ind2pool = {1: [2, 3, 4], 2 : [1, 3, 4], 3 : [1, 2, 4], 4 : [1, 2, 3]}

    def __init__(self, combine, **kwargs):
        self.combine = combine
        kwargs['dpp'] = False
        kwargs['theta'] = None
        super(AGNEWs_HotFlip_Targeted, self).__init__(**kwargs)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        X_init = X.clone()
        y = self.y[idx]
        y_rand = self.y_rand[idx]
        X = self.corrupt(X, (y_rand, y))
        return X_init, X, y_rand

    def load(self, label_data_path, lowercase = True):
        random.seed(42)
        self.label = []
        self.data = []
        self.label_rand = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]))
                self.label_rand.append(random.choice(self.ind2pool[self.label[-1]]))
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)
        self.y = torch.LongTensor(self.label)
        self.y_rand = torch.LongTensor(self.label_rand)
        
    def corrupt_candidates(self, X, ys):
        y_rand, y_true = ys
        inputs = Variable(torch.unsqueeze(X, 0).to("cuda:0"), requires_grad=True)
        target_rand = Variable(torch.unsqueeze(y_rand, 0).to("cuda:0")).sub(1)
        target_true = Variable(torch.unsqueeze(y_true, 0).to("cuda:0")).sub(1)
        self.model.eval()
        loss = -F.nll_loss(self.model(inputs), target_rand)
        if self.combine:
            loss += F.nll_loss(self.model(inputs), target_true)
        loss.backward()
        grads = inputs.grad.cpu().data.numpy()[0, ]
        new_data = X.data.numpy()
        x, y = np.nonzero(new_data)
        values = grads[:, y] - grads[x, y]
        values[x, np.arange(y.shape[0])] = -np.inf
        num_corrupt = int(round(y.shape[0] * self.per_corrupt / 100.))
        candidates = sorted(zip(values.max(axis=0), y, values.argmax(axis=0)), reverse=True)
        return candidates, new_data, num_corrupt