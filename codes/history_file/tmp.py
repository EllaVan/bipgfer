import sys
sys.path.append('/media/database/data4/wf/AU_EMOwPGM/codes')

import os
import csv
import pandas as pd
import pickle as pkl

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation

import numpy as np

from materials.process_priori import cal_interAUPriori
from models.AU_EMO_BP import UpdateGraph
from models.AU_EMO_bayes import initGraph, AU_EMO_bayesGraph
from models.RadiationAUs import RadiateAUs
import utils


def run_likePGM(args):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    c = list(map(int, AU))
    AU = c
    
    # 建立索引和EMO/AU名称的双向映射
    index2EMO_dict = dict(zip(range(len(EMO)), EMO))
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
    AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
    index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))

    if args.dataset == 'BP4D':
        path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'
        EMO_code_dict = {
            'T1': 'happy',
            'T2': 'sad',
            'T3': 'surprise',
            'T4': 'embarrassment',
            'T5': 'fear',
            'T6': 'physical pain',
            'T7': 'anger',
            'T8': 'disgust',
        }
    
    # 建立AU-EMO概率图和推理模型
    # AU_EMO_model = initGraph()
    AU_EMO_model = AU_EMO_bayesGraph(EMO2AU_cpt, prob_AU, EMO, AU)
    # model_infer = VariableElimination(AU_EMO_model)
    
    # 按训练数据集路径开始训练
    file_list = os.listdir(path_info)
    for file in file_list:
        cur_info_path = os.path.join(path_info, file)

        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]
        if task != 'T4' and task != 'T6': # 暂时不考虑基本表情以外的任务
            emo_name = EMO_code_dict[task] 
            emo_label = EMO2index_dict[emo_name]

            # 建立输出文件
            outfile = os.path.join(args.save_path, file.split('.')[0] + 'pred_PGM.csv')
            header = ['idx'] + EMO + ['Result='+emo_name]
            with open(outfile, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                    
            AU_evidence = {}
            cur_row = pd.read_csv(cur_info_path) # 读取文件内容
            cur_info = np.array(cur_row.iloc[:, 1:]) # 文件中AU的发生情况从第二列起，第一列为frame编码

            for frame in range(cur_info.shape[0]): # 遍历每一帧
                cur_item = np.array(cur_info[frame])
                for j in AU:
                    AU_evidence['AU'+str(j)] = cur_item[j-1] # 按照建模的AU填充AU_evidence，也就是发生的AU序列
                    if cur_item[j-1] == 9:
                        AU_evidence['AU'+str(j)] = 0 # 数据集中无标注的AU按未发生处理
                if emo_label == 0 : # 数据集中未标注的但是原先验中有的AU，按先验的给定数值，相当于固定这部分，不做更新
                    AU_evidence['AU26'] = 1
                elif emo_label == 2 or emo_label == 4:
                    AU_evidence['AU25'] = 1
                    AU_evidence['AU26'] = 1
                elif emo_label == 5:
                    AU_evidence['AU25'] = 1
                
                q = AU_EMO_model.infer(AU_evidence) # 按照标注的AU序列进行推理

                cur_idx = [cur_row.iloc[frame, 0]] # 当前帧的编码
                cur_prob = list(q.values) # 由概率图推理出的各EMO可能的概率
                cur_out = [index2EMO_dict[np.argmax(q.values)]] # 由PGM + AU序列推理出的可能的EMO状态号
                cur_pred = cur_idx + cur_prob + cur_out
                with open(outfile, 'a', encoding='UTF8', newline='') as f: # 写入PGM推理的结果
                    writer = csv.writer(f)
                    writer.writerow(cur_pred)

    end_flag = True


def run_myIdea(args):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    
    # 建立索引和EMO/AU名称的双向映射
    index2EMO_dict = dict(zip(range(len(EMO)), EMO))
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
    AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
    index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))

    path_info, EMO_code_dict = utils.getDatasetInfo(args.dataset)

    AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
    AU2EMO_cpt = []
    for i in range(len(AU)):
        AU2EMO_cpt.append(list(AU_EMO_jpt[:, i]/prob_AU[i]))

    # 交叉熵损失函数
    CE = nn.CrossEntropyLoss()
    
    # 按训练数据集路径开始训练
    file_list = os.listdir(path_info)
    end_flag = 0
    for file in file_list:
        end_flag += 1

        cur_info_path = os.path.join(path_info, file)
        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]
        if task != 'T4' and task != 'T6': # 暂时不考虑基本表情以外的任务
            emo_name = EMO_code_dict[task] 
            emo_label = EMO2index_dict[emo_name]

            # 建立输出文件
            log_dir = os.path.join(args.save_path, file.split('.')[0])
            err_writer = SummaryWriter(log_dir)
            outfile = os.path.join(log_dir, 'pred_Idea.csv')
            header = ['idx'] + EMO + ['Result='+emo_name]
            with open(outfile, 'w', encoding='UTF8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(header)     
            
            file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
            err_record = []
            for frame in range(file_info.shape[0]): # 遍历每一帧
                weight = []
                x = []
                cur_item = np.array(file_info[i])
                x = [1.0] * len(AU)
                for i, au in enumerate(AU):
                    if cur_item[au] == 0:
                        weight.append([1-value for value in list(EMO2AU_cpt[:, i])])
                    elif cur_item[au] == 1:
                        weight.append(list(EMO2AU_cpt[:, i]))

                if emo_label == 0:
                    weight.append([1-value for value in list(EMO2AU_cpt[:, i])])
                    weight.append(list(EMO2AU_cpt[:, i]))
                elif emo_label == 2 or emo_label == 4:
                    weight.append(list(EMO2AU_cpt[:, i]))
                    weight.append(list(EMO2AU_cpt[:, i]))
                elif emo_label == 5:
                    weight.append(list(EMO2AU_cpt[:, i]))
                    weight.append([1-value for value in list(EMO2AU_cpt[:, i])])


                update = UpdateGraph(len(AU), len(EMO), np.array(weight)).to(device)
                optim_graph = optim.SGD(update.parameters(), lr=0.0001)
                
                AU_evidence = torch.tensor(x, dtype=torch.float32).to(device)
                cur_prob = update(AU_evidence)
                err = CE(cur_prob.unsqueeze(0), torch.tensor(emo_label, dtype=torch.long).unsqueeze(0).to(device))
                err_record.append(utils.loss_to_float(err))
                
                cur_idx = [file_info[frame, 0]] # 当前帧的编码
                cur_out = [index2EMO_dict[int(torch.argmax(cur_prob))]] # 由PGM + AU序列推理出的可能的EMO状态号
                cur_pred = cur_idx + cur_prob.tolist() + cur_out
                with open(outfile, 'a', encoding='UTF8', newline='') as f: # 写入PGM推理的结果
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(cur_pred)

                err_writer.add_scalar('err', np.array(err.detach().cpu()), frame)
                optim_graph.zero_grad()
                err.backward()
                optim_graph.step()

                new_EMO2AU_cpt = []
                for k in range(len(EMO)):
                    if k == emo_label:
                        new_EMO2AU_cpt.append(update.fc.weight[k, :].detach().cpu().numpy())
                    else:
                        new_EMO2AU_cpt.append(EMO2AU_cpt[k, :])
                EMO2AU_cpt_tmp = np.array(new_EMO2AU_cpt)
                EMO2AU_cpt = np.where(EMO2AU_cpt_tmp >= 0, EMO2AU_cpt_tmp, 0)
                EMO2AU_cpt = np.where(EMO2AU_cpt_tmp <= 1, EMO2AU_cpt_tmp, 1)

        if end_flag == 8:
            break


def run_myIdea_v2(args):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    
    # 建立索引和EMO/AU名称的双向映射
    index2EMO_dict = dict(zip(range(len(EMO)), EMO))
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
    AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
    index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))

    path_info, EMO_code_dict = utils.getDatasetInfo(args.dataset)

    AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
    AU2EMO_cpt = []
    for i in range(len(AU)):
        AU2EMO_cpt.append(list(AU_EMO_jpt[:, i]/prob_AU[i]))
    AU2EMO_cpt = np.array(AU2EMO_cpt)

    # 交叉熵损失函数
    CE = nn.CrossEntropyLoss()
    
    # 按训练数据集路径开始训练
    file_list = os.listdir(path_info)
    end_flag = 0
    for file in file_list:
        end_flag += 1 # 对训练文件计数
        
        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]
        if task != 'T4' and task != 'T6': # 暂时不考虑基本表情以外的任务
            emo_name = EMO_code_dict[task] 
            emo_label = EMO2index_dict[emo_name]

            # 建立输出文件
            log_dir = os.path.join(args.save_path, file.split('.')[0])
            err_writer = SummaryWriter(log_dir)
            outfile = os.path.join(log_dir, 'pred_Idea.csv')
            header = ['idx'] + EMO + ['Result='+emo_name]
            with open(outfile, 'w', encoding='UTF8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(header)     
            
            cur_info_path = os.path.join(path_info, file)
            file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
            err_record = []
            for frame in range(file_info.shape[0]): # 遍历每一帧
                cur_item = np.array(file_info[frame, :])
                x = []
                weight1 = []
                weight2 = []
                
                for i, au in enumerate(AU):
                    if cur_item[au] == 1:
                        x.append(1.0)
                        weight1.append(EMO2AU_cpt[:, i])
                        weight2.append(AU2EMO_cpt[i, :])
                if emo_label == 0:
                    x.append(1.0)
                    weight1.append(EMO2AU_cpt[:, -2])
                    weight2.append(AU2EMO_cpt[-2, :])
                elif emo_label == 2 or emo_label == 4:
                    x.append(1.0)
                    x.append(1.0)
                    weight1.append(EMO2AU_cpt[:, -1])
                    weight1.append(EMO2AU_cpt[:, -2])
                    weight2.append(AU2EMO_cpt[-1, :])
                    weight2.append(AU2EMO_cpt[-2, :])
                elif emo_label == 5:
                    x.append(1.0)
                    weight1.append(EMO2AU_cpt[:, -1])
                    weight2.append(AU2EMO_cpt[-1, :])

                update = UpdateGraph(len(AU), len(EMO), EMO2AU_cpt).to(device)
                optim_graph = optim.SGD(update.parameters(), lr=0.0001)
                
                AU_evidence = torch.tensor(x, dtype=torch.float32).to(device)
                cur_prob = update(AU_evidence)

                err = CE(cur_prob.unsqueeze(0), torch.tensor(emo_label, dtype=torch.long).unsqueeze(0).to(device))
                err_record.append(utils.loss_to_float(err))
                
                cur_idx = [file_info[frame, 0]] # 当前帧的编码
                cur_pred = [index2EMO_dict[int(torch.argmax(cur_prob))]] # 由PGM + AU序列推理出的可能的EMO状态号
                cur_out = cur_idx + cur_prob.tolist() + cur_pred
                with open(outfile, 'a', encoding='UTF8', newline='') as f: # 写入PGM推理的结果
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(cur_out)

                err_writer.add_scalar('err', np.array(err.detach().cpu()), frame)
                optim_graph.zero_grad()
                err.backward()
                optim_graph.step()

                new_EMO2AU_cpt = []
                for k in range(len(EMO)):
                    if k == emo_label:
                        new_EMO2AU_cpt.append(update.fc.weight[k, :].detach().cpu().numpy())
                    else:
                        new_EMO2AU_cpt.append(EMO2AU_cpt[k, :])
                EMO2AU_cpt_tmp = np.array(new_EMO2AU_cpt)
                EMO2AU_cpt = np.where(EMO2AU_cpt_tmp >= 0, EMO2AU_cpt_tmp, 0)
                EMO2AU_cpt = np.where(EMO2AU_cpt_tmp <= 1, EMO2AU_cpt_tmp, 1)

        if end_flag == 8:
            break


class UpdateGraph(nn.Module):
    def __init__(self, in_channels, out_channels, W):
        super(UpdateGraph, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.tensor(W, dtype=torch.float32)

        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc.weight = Parameter(self.W.T)
        bn = nn.BatchNorm2d(num_features=out_channels, eps=0, affine=False, track_running_stats=False)

    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        self.out_prob = F.normalize(self.out, p = 1, dim=0)
        return self.out_prob


class UpdateGraph(nn.Module):
    def __init__(self, in_channels, out_channels, W):
        super(UpdateGraph, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.tensor(W, dtype=torch.float32)

        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc.weight = Parameter(self.W.T)

    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        self.out_prob = F.normalize(self.out, p = 1, dim=1)
        return self.out_prob


def initGraph():
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    c = list(map(int, AU))
    AU = c

    AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
    AU2EMO_cpt = []
    for i in range(len(AU)):
        AU2EMO_cpt.append(list(AU_EMO_jpt[:, i]/prob_AU[i]))
    
    EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, 1e-4)
    # 建立由EMO指向AU的概率图
    condition_edge = []
    EMO2_AU_edg = []
    for i in AU:
        condition_edge.append(('condition', 'AU'+str(i)))
        EMO2_AU_edg.append(('EMO', 'AU'+str(i)))
    edges = condition_edge + EMO2_AU_edg
    AU_EMO_model = BayesianNetwork(edges)
    # EMO的条件概率，由于EMO没有父亲节点，故值为EMO的边缘概率
    EMO_cpd = TabularCPD(variable='EMO', variable_card=len(EMO), values=[[1.0/len(EMO)]]*len(EMO))
    condition_cpd = TabularCPD(variable='condition', variable_card=2, values=[[0], [1]])
    AU_EMO_model.add_cpds(EMO_cpd)
    AU_EMO_model.add_cpds(condition_cpd)
    # 对于每一个AU
    AU_cpd = {}
    condition = {}
    for i, j in enumerate(AU):
        var = 'AU'+str(j)
        AU_cpd[j] = TabularCPD(variable=var, variable_card=2, 
                               values=[[1-value for value in list(EMO2AU_cpt[:, i])], list(EMO2AU_cpt[:, i])],
                               evidence=['EMO'], evidence_card=[len(EMO)]
                               ) # P(AU | EMO)
        condition[j] = TabularCPD(variable=var, variable_card=2, 
                               values=[[0.7, 0.3], [0.3 ,0.7]],
                               evidence=['condition'], evidence_card=[2]
                               ) # P(AU | EMO)
        AU_EMO_model.add_cpds(AU_cpd[j])
        AU_EMO_model.add_cpds(condition[j])
    AU_EMO_model.check_model() #确认模型是否正确
    return AU_EMO_model


def interAUGraph():
    AU = [1, 2, 3, 4]
    # 建立AU内部的概率图
    interAU = []
    for i in AU:
        for j in AU:
            if i < j:
                interAU.append(('AU'+str(i), 'AU'+str(j)))
    interAU_model = BayesianNetwork(interAU)
    # 对于每一个AU
    AU1_cpd = TabularCPD(variable='AU1', variable_card=2, values=[[0.6], [0.4]])
    interAU_model.add_cpds(AU1_cpd)
    for i, pair in enumerate(interAU):
        head = pair[0]
        tail = pair[1]
        AU_cpd_i = TabularCPD(variable=tail, variable_card=2, evidence=[head],evidence_card=[2],
                                values=[[0.7,0.2],[0.3,0.8]])
        interAU_model.add_cpds(AU_cpd_i)
    interAU_model.check_model() #确认模型是否正确
    return interAU_model


def run_tmp():
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

    from models.AU_EMO_BP import UpdateGraph, UpdateGraph_whole
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

        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--lr_decay_idx', type=int, default=5000)
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
        update = UpdateGraph_whole(device, in_channels=1, out_channels=len(EMO), W=EMO2AU_cpt, AUthresh=args.AUthresh).to(device)
        
        optim_graph = optim.SGD(update.parameters(), lr=args.lr)
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

            if len(occ_au) != 0:
                num_all_img += 1
                prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=args.AUthresh)
                
                pos = list(np.where(prob_all_au > args.AUthresh)[0]) # pos = np.where(prob_all_au == 1)[0]
                weight = np.array(weight)
                
                for i in range(prob_all_au.shape[0]-2):
                    if i in pos:
                        prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                    else:
                        prob_all_au[i] = 1 / (1-prob_AU[i])
                        weight[i, :] = 1 - weight[i, :]

                # prob_all_au[-2] = 1 / prob_AU[-2]
                # prob_all_au[-1] = 1 / prob_AU[-1]
                if emo_label == 0:
                    weight[-1, :] = 1 - weight[-1, :]
                    AU_cnt[-2] += 1
                    pos.append(len(AU)-2)
                    prob_all_au[-1] = 1 / (1-prob_AU[-1])
                    prob_all_au[-2] = 1 / prob_AU[-2]
                elif emo_label == 2 or emo_label == 4:
                    AU_cnt[-2] += 1
                    AU_cnt[-1] += 1
                    pos.append(len(AU)-2)
                    pos.append(len(AU)-1)
                    prob_all_au[-2] = 1 / prob_AU[-2]
                    prob_all_au[-1] = 1 / prob_AU[-1]
                else:
                    weight[-2, :] = 1 - weight[-2, :]
                    weight[-1, :] = 1 - weight[-1, :]
                    prob_all_au[-2] = 1 / (1-prob_AU[-2])
                    prob_all_au[-1] = 1 / (1-prob_AU[-1])
                        
                weight = np.where(weight > 0, weight, args.zeroPad)
                update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=weight, prob_all_au=prob_all_au).to(device)
                optim_graph = optim.SGD(update.parameters(), lr=args.lr)
                
                AU_evidence = torch.ones((1, 1)).to(device)
                cur_prob = update(AU_evidence)
                # cur_prob, pos = update(prob_all_au, prob_AU, emo_label)
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
                update_info2 = update.d1.detach().cpu().numpy().squeeze()
                # for emo_i, emo_name in enumerate(EMO):
                emo_i =emo_label
                for i, j in enumerate(AU):
                    factor = update_info2[emo_i] / weight[i, emo_i]
                    if i in pos:
                        if emo_i==emo_label:
                            new_EMO2AU_cpt[emo_i, i] -= args.lr*update_info1[emo_i]*factor
                        else:
                            tmp = weight.T[emo_i, i] - args.lr*update_info1[emo_i]*factor
                            new_EMO2AU_cpt[emo_i, i] = 1 - tmp
                    # else:
                    #     if emo_i==emo_label:
                    #         new_EMO2AU_cpt[emo_i, i] -= args.lr*update_info1[emo_i]*factor
                        '''
                        if i in pos:
                            if emo_i==emo_label:
                                new_EMO2AU_cpt[emo_i, i] += args.lr*np.abs(update_info1[emo_i]*factor)
                            else:
                                new_EMO2AU_cpt[emo_i, i] -= args.lr*np.abs(update_info1[emo_i]*factor)
                        else:
                            if emo_i==emo_label:
                                new_EMO2AU_cpt[emo_i, i] -= args.lr*np.abs(update_info1[emo_i]*factor)
                            else:
                                new_EMO2AU_cpt[emo_i, i] += args.lr*np.abs(update_info1[emo_i]*factor)
                        if i in pos:
                            new_EMO2AU_cpt[emo_label, i] += args.lr*np.abs(update_info1[emo_label]*factor)
                        else:
                            new_EMO2AU_cpt[emo_label, i] -= args.lr*np.abs(update_info1[emo_label]*factor)
                        '''
                        
                '''
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
                    new_EMO2AU_cpt[emo_label, -1] -= args.lr*np.abs(update_info1[emo_label]*factor)
                else:
                    factor = update_info2[emo_label] / weight[-2, emo_label]
                    new_EMO2AU_cpt[emo_label, -2] -= args.lr*np.abs(update_info1[emo_label]*factor)
                    factor = update_info2[emo_label] / weight[-1, emo_label]
                    new_EMO2AU_cpt[emo_label, -1] -= args.lr*np.abs(update_info1[emo_label]*factor)
                '''
                
                EMO2AU_cpt = np.array(new_EMO2AU_cpt)
                EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, 1e-4)
                EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

                for i, au_i in enumerate(pos): #occ_au
                    for j, au_j in enumerate(pos): #occ_au
                        if i != j:
                            AU_ij_cnt[au_i][au_j] += 1
                            AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
                # EMO2AU_cpt = update.W.data.detach().cpu().numpy()
                for i, j in enumerate(AU):
                    prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
                prob_AU = np.where(prob_AU > 0, prob_AU, args.zeroPad)
                prob_AU = np.where(prob_AU <= 1, prob_AU, 1)

                # if idx == args.lr_decay_idx:
                #     args.lr /= 10.0
                
        save_pkl['new_EMO2AU'] = EMO2AU_cpt
        save_pkl['new_AU_cpt'] = AU_cpt
        save_pkl['train_size'] = num_all_img - ori_size
        save_pkl['new_prob_AU'] = prob_AU
        save_pkl['AU_cnt'] = AU_cnt
        save_pkl['AU_ij_cnt'] = AU_ij_cnt
        with open(os.path.join(args.save_path, 'results.pkl'), 'wb') as fo:
            pkl.dump(save_pkl, fo)
        fo.close()
        end_flag = True 
    

def run_myIdea_v3(args):
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

        if len(occ_au) >= 3:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=args.AUthresh)
            
            pos = np.where(prob_all_au > args.AUthresh)[0] # pos = np.where(prob_all_au == 1)[0]
            weight = np.array(weight)
            
            prob_all_au[-2] = 1 / prob_AU[-2]
            prob_all_au[-1] = 1 / prob_AU[-1]
            if emo_label == 0:
                AU_cnt[-2] += 1
                weight[-1, :] = 1 - weight[-1, :]
            elif emo_label == 2 or emo_label == 4:
                AU_cnt[-2] += 1
                AU_cnt[-1] += 1
                # pass
            # elif emo_label == 5:
            #     weight[-2, :] = 1 - weight[-2, :]
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
            update_info2 = update.d1.detach().cpu().numpy().squeeze()
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
            # elif emo_label == 5:
            #     factor = update_info2[emo_label] / weight[-2, emo_label]
            #     new_EMO2AU_cpt[emo_label, -2] -= args.lr*np.abs(update_info1[emo_label]*factor)
            #     factor = update_info2[emo_label] / weight[-1, emo_label]
            #     new_EMO2AU_cpt[emo_label, -1] += args.lr*np.abs(update_info1[emo_label]*factor)
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


if __name__ == '__main__':
    global device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    AU_EMO_model = initGraph()

    model_infer = BeliefPropagation(AU_EMO_model)
    q = model_infer.query(variables=['EMO'], evidence={'condition': 1,
                                                    # 'EMO': 0,
                                                    # 'AU1':1, 
                                                    #  'AU2':0,
                                                    #  'AU4':0,
                                                    #  'AU5':0,
                                                    #  'AU6':0,
                                                    #  'AU7':0,
                                                    #  'AU9':0,
                                                    #  'AU10':0,
                                                    #  'AU11':0,
                                                    #  'AU12':0,
                                                    #  'AU15':0,
                                                    #  'AU17':1,
                                                    #  'AU20':0,
                                                    #  'AU23':0,
                                                    #  'AU24':0,
                                                    #  'AU25':0,
                                                    #  'AU26':0
                                                     })
    print(q)
