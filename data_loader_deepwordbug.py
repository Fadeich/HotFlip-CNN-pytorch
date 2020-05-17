#!/usr/bin/env python3
from data_loader import AGNEWs

import numpy as np
import torch.nn.functional as F
import torch

class WordBug(AGNEWs):
    def __init__(self, per_corrupt, model, scoring='greedy', **kwargs):
        self.per_corrupt = per_corrupt
        self.model = model
        self.classes = 4
        self.delta = 50
        self.scoring = scoring
        super(WordBug, self).__init__(**kwargs)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        X_init = X.clone()
        y = self.y[idx]
        X = self.corrupt(X, y)
        return X_init, X, y
    
    def corrupt(self, X, pred):
        new_data = X.data.numpy()
        x, y = np.nonzero(new_data)
        num_corrupt = int(round(y.shape[0] * self.per_corrupt / 100.))
        if num_corrupt == 0:
            return torch.from_numpy(new_data)
        candidates = self.corrupt_candidates(X, pred)
        if self.scoring == 'greedy':
            _, ind_y, new_pos, _ = zip(*candidates[:num_corrupt])
        elif self.scoring == 'beam':
            ind_y, new_pos = self.best_first_search(candidates, new_data, pred, num_corrupt)
        else:
            raise RuntimeError('scoring %s is invalid' % self.scoring)
        new_val = np.zeros((len(self.alphabet), num_corrupt))
        new_val[new_pos, np.arange(num_corrupt)] = 1
        new_data[:, ind_y] = new_val
        return torch.from_numpy(new_data)
    
    def best_first_search(self, candidates, new_data, pred, num_corrupt):
        score, y_ind, new_char, dict_ = candidates[0]
        best_candidate = (score, [y_ind], [new_char], dict_)
        for i in range(num_corrupt-1):
            data = new_data.copy()
            n = len(best_candidate[1])
            new_val = np.zeros((len(self.alphabet), n))
            new_val[best_candidate[2], np.arange(n)] = 1
            data[:, best_candidate[1]] = new_val
            next_best_candidate = self.corrupt_candidates(torch.from_numpy(data), pred, 
                                                          best_candidate[1][-1], best_candidate[3])[0]
            best_candidate = (best_candidate[0] + next_best_candidate[0],
                              best_candidate[1] + [next_best_candidate[1]],
                              best_candidate[2] + [next_best_candidate[2]],
                              next_best_candidate[3])
        return best_candidate[1], best_candidate[2]
    
    def corrupt_candidates(self, X, pred, prev_position=None, scores_dict=None):
        dloss, poses = self.replaceone(X, pred-1, prev_position, scores_dict)
        n = len(dloss)
        np.random.seed(42)
        new_pos = np.random.randint(1, 70, size=n)
        scores_dict = [dict(zip(poses, dloss))]*len(poses)
        return sorted(zip(dloss, poses, new_pos, scores_dict), reverse=True)
    
    def replaceone(self, inputs, pred, prev_position, scores_dict):
        losses = []
        poses = []
        losses_init = []
        poses_init = []
        for i in range(inputs.size()[1]):
            if not inputs[:, i].sum():
                continue
            if scores_dict is not None and (i < prev_position - self.delta or i > prev_position + self.delta):
                losses_init.append(scores_dict[i])
                poses_init.append(i)
                continue
            tempinputs = inputs.clone()
            tempinputs[:,i] = 0
            inp = torch.unsqueeze(tempinputs, 0).to("cuda:0")
            with torch.no_grad():
                tempoutput = self.model(inp)
            loss = F.nll_loss(tempoutput, pred.reshape((1,)).cuda())
            losses.append(loss.cpu().numpy())
            poses.append(i)
        losses = np.array(losses)
        losses /= np.abs(np.sum(losses))
        if scores_dict is not None:
            losses *= losses.shape[0]/len(scores_dict)
        return list(losses) + losses_init, poses + poses_init