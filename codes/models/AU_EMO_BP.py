import sys
sys.path.append('/media/database/data4/wf/AU_EMOwPGM/codes')

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

from models import AU_EMO_bayes
from materials.process_priori import cal_interAUPriori

AU_evidence = { 'AU1':1, 
                    'AU2':0,
                    'AU4':1,
                    'AU5':0,
                    'AU6':0,
                    'AU7':1,
                    'AU9':1,
                    'AU10':0,
                    'AU11':0,
                    'AU12':0,
                    'AU15':0,
                    'AU17':0,
                    'AU20':0,
                    'AU23':0,
                    'AU24':0,
                    'AU25':1,
                    'AU26':1 }


class UpdateGraph(nn.Module):
    def __init__(self, device, in_channels=1, out_channels=6, W=None, prob_all_au=None, init=None):
        super(UpdateGraph, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.W = torch.tensor(W, dtype=torch.float32).to(self.device)
        self.W = Parameter(self.W)
        if prob_all_au is None:
            prob_all_au = [[1]] * W.shape[0]
        self.prob_all_au = Parameter(torch.tensor(prob_all_au, dtype=torch.float32).to(self.device))
        
        if init is None:
            self.d = torch.ones((self.in_channels, self.out_channels)).to(self.device)
        else:
            self.init = init
            self.d = self.init.clone().to(self.device)
        for i in range(self.W.shape[1]):
            for j in range(self.W.shape[0]):
                self.d[:, i] = self.d[:, i]*self.W[j][i]*self.prob_all_au[j]
        self.d1 = F.normalize(self.d, p = 1, dim=1).to(self.device)
        self.fc = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.fc.weight = Parameter(self.d1.T)

    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        return self.out


class UpdateGraph_whole(nn.Module):
    def __init__(self, device, in_channels=1, out_channels=6, W=None, AUthresh=0.6): # W->EMO2AU
        super(UpdateGraph_whole, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.AUthresh = AUthresh
        W = np.where(W > 0, W, 1e-4)
        self.W = torch.tensor(W, dtype=torch.float32)
        # self.W = torch.where(self.W > 0, self.W, torch.tensor(1e-4))
        self.W = Parameter(self.W, requires_grad=True)

    def forward(self, prob_all_au, prob_AU, emo_label):
        if prob_all_au is None:
            prob_all_au = [[1]] * self.W.shape[0]
        pos = list(np.where(prob_all_au > self.AUthresh)[0]) # pos = np.where(prob_all_au == 1)[0]
        self.prob_all_au = torch.tensor(prob_all_au, dtype=torch.float32).to(self.device)
        
        weight = torch.rand((self.W.shape[1], self.W.shape[0])).to(self.device)
        for i in range(prob_all_au.shape[0]-2):
            if i in pos:
                self.prob_all_au[i] = self.prob_all_au[i] / prob_AU[i]
                weight[i, :] = self.W.T[i, :]
            else:
                self.prob_all_au[i] = 1 / (1-prob_AU[i])
                weight[i, :] = 1 - self.W.T[i, :]

        if emo_label == 0:
            weight[-1, :] = 1 - self.W.T[-1, :]
            weight[-2, :] = self.W.T[-2, :]
            pos.append(self.W.shape[1]-2)
            self.prob_all_au[-1] = 1 / (1-prob_AU[-1])
            self.prob_all_au[-2] = 1 / prob_AU[-2]
        elif emo_label == 2 or emo_label == 4:
            weight[-2, :] = self.W.T[-2, :]
            weight[-1, :] = self.W.T[-1, :]
            pos.append(self.W.shape[1]-2)
            pos.append(self.W.shape[1]-1)
            self.prob_all_au[-2] = 1 / prob_AU[-2]
            self.prob_all_au[-1] = 1 / prob_AU[-1]
        else:
            weight[-2, :] = 1 - self.W.T[-2, :]
            weight[-1, :] = 1 - self.W.T[-1, :]
            self.prob_all_au[-2] = 1 / (1-prob_AU[-2])
            self.prob_all_au[-1] = 1 / (1-prob_AU[-1])

        self.d = torch.ones((self.in_channels, self.out_channels)).to(self.device)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.d[:, i] *= weight[j][i]*self.prob_all_au[j]
        self.d1 = F.normalize(self.d, p = 1, dim=1)
        # self.fc = nn.Linear(self.in_channels, self.out_channels, bias=False)
        # self.fc.weight.data = self.d1.data.T

        # self.x = torch.ones((1, 1)).to(self.device)
        # self.out = self.fc(self.x)
        return self.d1, pos


if __name__ == '__main__':
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    AU_EMO_model = AU_EMO_bayes.AU_EMO_bayesGraph(EMO2AU_cpt, prob_AU, EMO, AU)
    q = AU_EMO_model.infer(AU_evidence)

    update_graph = UpdateGraph(AU, AU_evidence, EMO2AU_cpt)
    q1 = update_graph()

    end_flag = True
