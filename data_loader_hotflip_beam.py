#!/usr/bin/env python3
from data_loader import AGNEWs

import numpy as np
from termcolor import colored
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

def dpp_next_element(kernel_matrix, R):
    max_length = len(R) + 1
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    for k, selected_item in enumerate(R):
        ci_optimal = cis[:k, selected_item]
        di_optimal = np.sqrt(di2s[selected_item] + 1e-8)
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
    return di2s

def dpp_candidates(candidates, y_inds, theta):
    n = len(candidates)
    scores, y_ind, new_char = zip(*candidates)
    R = [y_ind.index(i) for i in y_inds]
    y_ind = np.array(y_ind)
    scores = np.array(scores) + 1.
    alpha = theta / 2 / (1 - theta)
    deltas = np.absolute(y_ind[:, np.newaxis] - y_ind[np.newaxis, :]) * 1.
    np.fill_diagonal(deltas, 1)
    S = 1 / deltas
    new_scores = np.exp(alpha * scores)
    L = new_scores[:, np.newaxis].dot(new_scores[np.newaxis, :]) * S
    values = dpp_next_element(L, R)
    return sorted(list(zip(values, y_ind, new_char)), reverse=True)

class AGNEWs_HotFlip_Beam(AGNEWs):
    def __init__(self, per_corrupt, model, theta, k, dpp, **kwargs):
        self.per_corrupt = per_corrupt
        self.model = model
        self.theta = theta
        self.k = k
        self.dpp = dpp
        super(AGNEWs_HotFlip_Beam, self).__init__(**kwargs)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        X_init = X.clone()
        y = self.y[idx]
        X = self.corrupt(X, y)
        return X_init, X, y
    
    def corrupt(self, X, y):
        candidates, new_data, num_corrupt = self.corrupt_candidates(X, y)
        if self.dpp:
            candidates = dpp_candidates(candidates, [], self.theta)
        if num_corrupt == 0:
            return torch.from_numpy(new_data)
        ind_y, new_pos = self.beam_search(candidates, new_data, y, num_corrupt)
        new_val = np.zeros((len(self.alphabet), num_corrupt))
        new_val[new_pos, np.arange(num_corrupt)] = 1
        new_data[:, ind_y] = new_val
        return torch.from_numpy(new_data)
    
    def beam_search(self, candidates, new_data, y, num_corrupt):
        memory = [(score, [y_ind], [new_char]) for score, y_ind, new_char in candidates]
        for i in range(num_corrupt-1):
            new_memory = []
            for current_score, y_inds, new_chars in memory[:self.k]:
                data = new_data.copy()
                new_val = np.zeros((len(self.alphabet), len(y_inds)))
                new_val[new_chars, np.arange(len(y_inds))] = 1
                data[:, y_inds] = new_val
                candidates_step, _, _ = self.corrupt_candidates(torch.from_numpy(data), y)
                if self.dpp:
                    candidates_step = dpp_candidates(candidates_step, y_inds, self.theta)
                for score_step, y_ind_step, new_char_step in candidates_step[:self.k]:
                    new_memory.append((current_score + score_step, y_inds + [y_ind_step], new_chars + [new_char_step]))
            memory = sorted(new_memory, reverse=True)
        return memory[0][1], memory[0][2]
        
    def corrupt_candidates(self, X, y):
        inputs = Variable(torch.unsqueeze(X, 0).to("cuda:0"), requires_grad=True)
        target = Variable(torch.unsqueeze(y, 0).to("cuda:0"))
        self.model.eval()
        loss = F.nll_loss(self.model(inputs), target.sub(1))
        loss.backward()
        grads = inputs.grad.cpu().data.numpy()[0, ]
        new_data = X.data.numpy()
        x, y = np.nonzero(new_data)
        values = grads[:, y] - grads[x, y]
        values[x, np.arange(y.shape[0])] = -np.inf
        num_corrupt = int(round(y.shape[0] * self.per_corrupt / 100.))
        candidates = sorted(zip(values.max(axis=0), y, values.argmax(axis=0)), reverse=True)
        return candidates, new_data, num_corrupt
    
    def print_string(self, idx):
        print('original: {}'.format(self.data[idx]))
        lens = map(len, self.data[idx].split())
        X = self.oneHotEncode(idx)
        X_cloned = X.clone()
        X = self.corrupt(X, self.y[idx])
        changed_pos = np.nonzero((X_cloned != X).sum(0).numpy() != 0)[0]
        ind_x, ind_y = np.nonzero(X.numpy().T)
        changed = max(ind_x) - changed_pos
        result = [' ']*max(ind_x+1)
        for i, j in zip(ind_x, ind_y):
            result[i] = self.alphabet[j]
        print(colored('corrupt: ', 'red'), end='')
        for i, ch in enumerate(result[::-1]):
            if i in changed:
                print(colored(ch, 'red'), end='')
            else:
                print(ch, end='')
    
    def mean_dist(self, ind):
        X = self.oneHotEncode(ind)
        y = self.y[ind]
        candidates, new_data, num_corrupt = self.corrupt_candidates(X, y)
        if self.dpp:
            candidates = dpp_candidates(candidates, [], self.theta)
        if num_corrupt <= 1:
            return 0
        ind_y, new_pos = self.beam_search(candidates, new_data, y, num_corrupt)
        return np.mean(np.diff(np.sort(ind_y)))