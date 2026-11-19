import argparse
import datetime
from errno import EMULTIHOP
import pytz
import os
import shutil
import pickle as pkl

import csv
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.utils import get_example_model

from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs, interAUs
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='BP4D')
    parser.add_argument('--save_path', type=str, default='save')

    # --------------settings for feature extraction-------------
    parser.add_argument('--manualSeed', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)
    parser.add_argument('--save_epoch', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_idx', type=int, default=20000)
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-4)
    
    return parser.parse_args()


def run_myIdea(args):
    train_loader, test_loader = utils.getDatasetInfo(args.dataset)
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())

    summary_writer = SummaryWriter(args.save_path)

    nodes = ['AU'+str(au) for au in AU]
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_cpt = AU_cpt - np.eye(len(AU))
    AU_ij_cnt = AU_cpt * num_all_img
    AU_cnt = prob_AU * num_all_img

    save_pkl = {}
    save_pkl['ori_EMO2AU'] = EMO2AU_cpt
    save_pkl['ori_AU_cpt'] = AU_cpt
    save_pkl['ori_prob_AU'] = prob_AU

    # 交叉熵损失函数
    CE = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []

    for idx, (cur_item, emo_label, index) in enumerate(train_loader, 1):
        variable = nodes.copy()
        weight = []
        occ_au = []
        evidence = {}
        prob_all_au = np.zeros((len(AU),))
        
        cur_item = np.where(cur_item != 9, cur_item, 0)
        for i, au in enumerate(AU):
            if cur_item[0, au] == 1:
                occ_au.append(i)
                prob_all_au[i] = 1
                AU_cnt[i] += 1
                evidence['AU'+str(au)] = 1
                variable.remove('AU'+str(au))
            weight.append(EMO2AU_cpt[:, i])

        if len(occ_au) >= 2:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=args.AUthresh)
            
            pos = np.where(prob_all_au > args.AUthresh)[0] # pos = np.where(prob_all_au == 1)[0]
            weight = np.array(weight)
            
            prob_all_au[-2] = 1 / prob_AU[-2]
            prob_all_au[-1] = 1 / prob_AU[-1]
            if emo_label == 0:
                weight[-1, :] = 1 - weight[-1, :]
            elif emo_label == 2 or emo_label == 4:
                pass
            elif emo_label == 5:
                weight[-2, :] = 1 - weight[-2, :]
            else:
                weight[-2, :] = 1 - weight[-2, :]
                weight[-1, :] = 1 - weight[-1, :]
            
            for i in range(prob_all_au.shape[0]-2):
                if i in pos:
                    prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                else:
                    prob_all_au[i] = 1 / prob_AU[i]
                    weight[i, :] = 1 - weight[i, :]
                    
            weight = np.where(weight > 0, weight, args.zeroPad)
            update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=weight, prob_all_au=prob_all_au).to(device)
            optim_graph = optim.SGD(update.parameters(), lr=args.lr)
            
            AU_evidence = torch.ones((1, 1)).to(device)
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)

            err = CE(cur_prob, emo_label.to(device))
            err_record.append(utils.loss_to_float(err))
            acc_record.append(torch.eq(cur_pred, emo_label.to(device)).type(torch.FloatTensor).item())

            summary_writer.add_scalar('err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('err_single', utils.loss_to_float(err), idx)
            summary_writer.add_scalar('acc', np.array(acc_record).mean(), idx)

            optim_graph.zero_grad()
            err.backward()
            optim_graph.step()
            
            new_EMO2AU_cpt = EMO2AU_cpt
            update_info1 = update.fc.weight.grad.cpu().numpy()
            update_info2 = update.d1.cpu().numpy().squeeze()
            for i, j in enumerate(AU[:-2]):
                factor = update_info2[emo_label] / weight[i, emo_label]
                if i in pos:
                    new_EMO2AU_cpt[emo_label, i] += args.lr*np.abs(update_info1[emo_label]*factor)
                else:
                    new_EMO2AU_cpt[emo_label, i] -= args.lr*np.abs(update_info1[emo_label]*factor)

            if emo_label == 0:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] += args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] -= args.lr*np.abs(update_info1[emo_label]*factor)
            elif emo_label == 2 or emo_label == 4:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] += args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] += args.lr*np.abs(update_info1[emo_label]*factor)
            elif emo_label == 5:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] -= args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] += args.lr*np.abs(update_info1[emo_label]*factor)
            else:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] -= args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] -= args.lr*np.abs(update_info1[emo_label]*factor)
            
            EMO2AU_cpt = np.array(new_EMO2AU_cpt)
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, 1e-4)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] += 1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(AU):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            prob_AU = np.where(prob_AU > 0, prob_AU, args.zeroPad)
            prob_AU = np.where(prob_AU <= 1, prob_AU, 1)

            if idx == args.lr_decay_idx:
                args.lr /= 10.0

    save_pkl['new_EMO2AU'] = EMO2AU_cpt
    save_pkl['new_AU_cpt'] = AU_cpt
    save_pkl['train_size'] = len(train_loader)
    save_pkl['new_prob_AU'] = prob_AU
    save_pkl['AU_cnt'] = AU_cnt
    save_pkl['AU_ij_cnt'] = AU_ij_cnt
    with open(os.path.join(args.save_path, 'results.pkl'), 'wb') as fo:
        pkl.dump(save_pkl, fo)
    fo.close()
    end_flag = True 
    

def test():
    args = parse_args()
    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = utils.getDatasetInfo(args.dataset)
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())

    nodes = ['AU'+str(au) for au in AU]
    AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
    AU2EMO_cpt = []
    for i in range(len(AU)):
        AU2EMO_cpt.append(list(AU_EMO_jpt[:, i]/prob_AU[i]))
    AU2EMO_cpt = np.array(AU2EMO_cpt)
    AU2EMO_cpt = np.where(AU2EMO_cpt >= 0, AU2EMO_cpt, 1e-4)

    base_path = '/media/database/data4/wf/AU_EMOwPGM/codes/save'
    task_path = '2022-07-02'
    base_path = os.path.join(base_path, task_path)
    pkl_path = os.path.join(base_path, 'results.pkl')
    if os.path.getsize(pkl_path) > 0: 
        with open(pkl_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
    ori_EMO2AU = pkl_file['ori_EMO2AU']
    ori_AU_cpt = pkl_file['ori_AU_cpt']
    ori_prob_AU = pkl_file['ori_prob_AU']
    new_EMO2AU = pkl_file['new_EMO2AU']
    new_AU_cpt = pkl_file['new_AU_cpt']
    new_prob_AU = pkl_file['new_prob_AU']

    oriACC = []
    newACC = []
    ori_class_acc = []
    new_class_acc = []
    
    for k, emo in enumerate(EMO):
        test_data = test_loader[k]
        ori_acc = []
        new_acc = []
        for idx, (cur_item, emo_label, index) in enumerate(test_loader[k], 1):
            variable = nodes.copy()
            ori_weight = []
            new_weight = []
            occ_au = []
            abs_au = []
            prob_all_au = np.zeros((len(AU),))
            evidence = {}
            
            for i, au in enumerate(AU):
                if cur_item[0, au] == 1:
                    prob_all_au[i] = 1
                    evidence['AU'+str(au)] = 1
                    occ_au.append(i)
                    evidence['AU'+str(au)] = 1
                    variable.remove('AU'+str(au))
                ori_weight.append(ori_EMO2AU[:, i])
                new_weight.append(new_EMO2AU[:, i])

            if len(occ_au) != 0:
                ori_prob_all_au = prob_all_au
                new_prob_all_au = prob_all_au

                new_prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=args.AUthresh)

                ori_pos = np.where(ori_prob_all_au > args.AUthresh)[0]
                new_pos = np.where(new_prob_all_au > args.AUthresh)[0]
                ori_weight = np.array(ori_weight)
                new_weight = np.array(new_weight)
                
                ori_prob_all_au[-2] = 1 / ori_prob_AU[-2]
                ori_prob_all_au[-1] = 1 / ori_prob_AU[-1]
                new_prob_all_au[-2] = 1 / new_prob_AU[-2]
                new_prob_all_au[-1] = 1 / new_prob_AU[-1]
                if emo_label == 0:
                    ori_weight[-1, :] = 1 - ori_weight[-1, :]
                    new_weight[-1, :] = 1 - new_weight[-1, :]
                elif emo_label == 2 or emo_label == 4:
                    pass
                elif emo_label == 5:
                    ori_weight[-2, :] = 1 - ori_weight[-2, :]
                    new_weight[-2, :] = 1 - new_weight[-2, :]
                else:
                    ori_weight[-2, :] = 1 - ori_weight[-2, :]
                    ori_weight[-1, :] = 1 - ori_weight[-1, :]
                    new_weight[-2, :] = 1 - new_weight[-2, :]
                    new_weight[-1, :] = 1 - new_weight[-1, :]
                
                for i in range(ori_prob_all_au.shape[0]-2):
                    if i in ori_pos:
                        ori_prob_all_au[i] = ori_prob_all_au[i] / ori_prob_AU[i]
                    else:
                        ori_prob_all_au[i] = 1 / ori_prob_AU[i]
                        ori_weight[i, :] = 1 - ori_weight[i, :]
                for i in range(new_prob_all_au.shape[0]-2):
                    if i in new_pos:
                        new_prob_all_au[i] = new_prob_all_au[i] / new_prob_AU[i]
                    else:
                        new_prob_all_au[i] = 1 / new_prob_AU[i]
                        new_weight[i, :] = 1 - new_weight[i, :]

                ori_update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=ori_weight, prob_all_au=ori_prob_all_au).to(device)
                new_update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=new_weight, prob_all_au=new_prob_all_au).to(device)

                AU_evidence = torch.ones((1, 1)).to(device)
                ori_prob = ori_update(AU_evidence)
                ori_pred = torch.argmax(ori_prob)
                new_prob = new_update(AU_evidence)
                new_pred = torch.argmax(new_prob)

                ori_acc.append(torch.eq(ori_pred, emo_label.to(device)).type(torch.FloatTensor).item())
                oriACC.append(torch.eq(ori_pred, emo_label.to(device)).type(torch.FloatTensor).item())
                new_acc.append(torch.eq(new_pred, emo_label.to(device)).type(torch.FloatTensor).item())
                newACC.append(torch.eq(new_pred, emo_label.to(device)).type(torch.FloatTensor).item())

        ori_class_acc.append(np.array(ori_acc).mean())
        new_class_acc.append(np.array(new_acc).mean())
        print('The ori Acc of %s is %.5f, and the new Acc is %.5f' 
                %(emo, np.array(ori_acc).mean(), np.array(new_acc).mean()))
    save_pkl = {}
    save_pkl['ori_class_acc'] = ori_class_acc
    save_pkl['oriACC'] = np.array(oriACC).mean()
    save_pkl['new_class_acc'] = new_class_acc
    save_pkl['newACC'] = np.array(newACC).mean()
    with open(os.path.join(base_path, 'TestResults.pkl'), 'wb') as fo:
        pkl.dump(save_pkl, fo)
    fo.close()
    print('The ori Acc of the dataset is %.5f, and the new Acc is %.5f' 
            %(np.array(oriACC).mean(), np.array(newACC).mean()))


if __name__ == '__main__':
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)

    args = parse_args()

    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print('using gpu:', args.gpu)

    cur_day = str(cur_time).split(' ')
    cur_day = cur_day[0]
    args.save_path = os.path.join(args.save_path, cur_day)
    if os.path.exists(args.save_path) is True:
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)
    else:
        os.makedirs(args.save_path)

    print('%s based Update' % (args.dataset))
    run_myIdea(args)
    test()
    