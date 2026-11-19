'''
CASME的EMO顺序为'happy', 'sad', 'fear', 'surprise', 'disgust'#, 'repression'
顺序和priori一致
'''

import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

import torch

from models.RadiationAUs import RadiateAUs, interAUs
from rules_learning.dataset_process import info_load
from models.AU_EMO_BP import UpdateGraph
import rules_learning.utils as utils

def comput_stat():
    train_loader, test_loader = info_load.getCASMEdata()
    _, _, _, _, EMO, AU = tuple(train_loader.dataset.priori.values())
    EMO = ['happy', 'sad', 'fear', 'surprise', 'disgust']#, 'repression']
    num_AU = len(AU)
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index

    path_info = '/media/data1/wf/AU_EMOwPGM/codes/dataset/CASME/CASME2-coding.xlsx'
    df = pd.read_excel(path_info)
    new_col = list(range(df.shape[1]))
    df.columns = new_col
    df = df.drop(2, axis=1)
    df = df.drop(6, axis=1)
    df_data = df.iloc[:, 1]
    df_AU = df.iloc[:, 5]
    df_label = df.iloc[:, 6]
    df = pd.concat([df_data, df_AU, df_label], axis=1)
    new_col = list(range(df.shape[1]))
    df.columns = new_col
    
    datalist = {}
    for k, emo in enumerate(EMO):
        datalist[k] = []

    AU_array = []
    for i in range(df.shape[0]):
        item = df.iloc[i, :]
        item_label = item[2]
        if item_label != 'others' and item_label != 'repression' and item_label != 'neutral':
        # if item_label != 'others' and item_label != 'neutral':
            AU_array_tmp = np.zeros((1, 99))
            label = EMO2index_dict[item[2]]
            AU_list = str(item[1]).split('+')
            for au in AU_list:
                if au[0] == 'R' or au[0] == 'L':
                    au = au[1:]
                au = int(au)
                AU_array_tmp[0, au] = 1
            AU_array = AU_array_tmp[0, AU]
            datalist[label].append(list(AU_array))

    img_num = []
    sta_EMO2AU_cpt = np.zeros((len(EMO), num_AU))
    sta_AU_cpt = np.zeros((num_AU, num_AU))
    sta_prob_AU = np.zeros((num_AU,))

    for i in range(len(datalist)):
        # EMO2AU_cpt
        EMO_AU = datalist[i]
        EMO_sumAU = np.sum(EMO_AU, axis=0)
        pos = list(np.where(EMO_sumAU > 0)[0])
        for j in pos:
            sta_EMO2AU_cpt[i][j] = EMO_sumAU[j]
            sta_prob_AU[j] += EMO_sumAU[j]
        img_num.append(len(EMO_AU))
        sta_EMO2AU_cpt[i] = sta_EMO2AU_cpt[i] / img_num[i]

        # interAU
        for j in range(len(EMO_AU)):
            item = np.array(EMO_AU[j])
            pos = list(np.where(item > 0)[0])
            if len(pos) != 1:
                for k1 in pos:
                    for k2 in pos:
                        if k1 != k2:
                            sta_AU_cpt[k1][k2] += 1
    dataset_size = np.sum(img_num)
    for i in range(num_AU):
        if sta_prob_AU[i] != 0:
            sta_AU_cpt[:, i] = sta_AU_cpt[:, i] / sta_prob_AU[i]
    sta_prob_AU = sta_prob_AU / dataset_size

    data_pkl = {}
    data_pkl['EMO'] = EMO
    data_pkl['AU'] = AU
    data_pkl['sta_AU_cpt'] = sta_AU_cpt
    data_pkl['sta_prob_AU'] = sta_prob_AU
    data_pkl['sta_EMO2AU_cpt'] = sta_EMO2AU_cpt
    data_pkl['img_num'] = img_num
    with open('/media/data1/wf/AU_EMOwPGM/codes/save/CASME/stastics/stat_nR.pkl', 'wb') as fo:
        pkl.dump(data_pkl, fo)
    end_flag = 0


def infer():
    pkl_path = '/media/data1/wf/AU_EMOwPGM/codes/save/CASME/statistics/stat_nR.pkl'
    with open(pkl_path, 'rb') as fo:
        data_pkl = pkl.load(fo)
    EMO = data_pkl['EMO']
    AU = data_pkl['AU']

    prob_AU = data_pkl['sta_prob_AU'][AU]
    EMO2AU_cpt = data_pkl['sta_EMO2AU_cpt'][:, AU]
    AU_cpt = data_pkl['sta_AU_cpt'][:, AU]
    AU_cpt = AU_cpt[AU, :]

    num_EMO = len(EMO)
    confu_m = torch.zeros((len(EMO), len(EMO)))
    _, test_loader = info_load.getCASMEdata(all_test=True)
    acc_all = []
    if_bonus = True
    for k, emo in enumerate(EMO):
        acc_record = []
        for idx, (cur_item, emo_label, index) in enumerate(test_loader[k], 1):
            weight = []
            occ_au = []
            prob_all_au = np.zeros((len(AU),))
            if len(cur_item.shape) > 2:
                cur_item = cur_item.squeeze(0)
            for i, au in enumerate(AU):
                if cur_item[0, au] == 1:
                    occ_au.append(i)
                    prob_all_au[i] = 1
                weight.append(EMO2AU_cpt[:, i])

            EMO_bonus = np.ones(len(EMO), )
            if len(occ_au) != 0:
                prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
                pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
                weight = np.array(weight)
                
                for i in range(prob_all_au.shape[0]):
                    if i in pos:
                        if i in occ_au:
                            prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                        else:
                            prob_all_au[i] = prob_all_au[i] / prob_AU[i] / num_EMO
                        if if_bonus is True and i in occ_au:
                            AU2EMO_bonus = np.ones(len(EMO), )
                            latent_pos = weight[i].argsort()
                            for j, bonus_emo in enumerate(latent_pos):
                                if j == 0:
                                    AU2EMO_bonus[bonus_emo] = 1
                                else:
                                    if weight[i][latent_pos[j]] == weight[i][latent_pos[j-1]]:
                                        AU2EMO_bonus[bonus_emo] = AU2EMO_bonus[latent_pos[j-1]]
                                    else:
                                        AU2EMO_bonus[bonus_emo] = j+1
                            EMO_bonus = EMO_bonus * AU2EMO_bonus
                    else:
                        prob_all_au[i] = 1 / (1-prob_AU[i])
                        weight[i, :] = 1 - weight[i, :]
                weight = np.where(weight > 0, weight, 1e-5)
                EMO_bonus = torch.Tensor(EMO_bonus / num_EMO).reshape(1, -1)
                
                update = UpdateGraph(device, in_channels=1, out_channels=len(EMO), W=weight, 
                                    prob_all_au=prob_all_au).to(device)
                
                AU_evidence = torch.ones((1, 1)).to(device)
                cur_prob = update(AU_evidence)
                bonus_prob = cur_prob * EMO_bonus.to(device)
                cur_pred = torch.argmax(bonus_prob)
                # confu_m = utils.confusion_matrix(cur_pred, labels=emo_label, conf_matrix=confu_m)
                confu_m[cur_pred, emo_label] += 1
                cur_p = torch.eq(cur_pred, emo_label.to(device)).type(torch.FloatTensor).item()
                acc_record.append(cur_p)
        print('The Acc of %s is %.5f' %(emo, np.array(acc_record).mean()))
        acc_all += acc_record
    print('The Acc of dataset is %.5f' %(np.array(acc_all).mean()))
    # utils.plot_confusion_matrix(confu_m.numpy(), classes=EMO, normalize=True,
    #                              title='Normalized confusion matrix')

    end_flag = 0


if __name__ == '__main__':
    global device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    comput_stat()
    end_flag = True