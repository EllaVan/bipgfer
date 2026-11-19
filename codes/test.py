import argparse
import enum
import os
import numpy as np
import pickle as pkl
from materials.process_priori import cal_interAUPriori
from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs
import torch
import utils
import pandas as pd
from pgmpy.inference import VariableElimination

def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='BP4D')
    parser.add_argument('--AUthresh', type=float, default=0.6)
    return parser.parse_args()


if __name__ == '__main__':
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
    task_path = '2022-07-19'
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

    '''
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
    values = list(EMO_code_dict.values())
    keys = list(EMO_code_dict.keys())
    index2EMO_dict = dict(zip(range(len(EMO)), EMO))
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
    index2EMO_dataset_dict = dict(zip(range(len(values)), values))
    EMO2index_dataset_dict = dict(zip(values, range(len(values)))) #通过名称找到index
    
    ori_EMO2AU = np.zeros_like(new_EMO2AU)
    df = pd.read_csv('/media/database/data4/wf/AU_EMOwPGM/codes/save/stastics/stastics_EMO2AU(occAU>1).csv', header=None).iloc[:, 0:]
    a = df.to_numpy()
    a = a[:, AU]
    for emo_i, emo_name in enumerate(EMO):
        i = EMO2index_dataset_dict[emo_name]
        ori_EMO2AU[emo_i, :] = a[i, :]
    ori_AU_cpt = pkl_file['new_AU_cpt']
    ori_prob_AU = pkl_file['ori_prob_AU']
    for i, j in enumerate(AU):
        ori_prob_AU[i] = np.sum(ori_EMO2AU[:, i]) / (len(EMO))
    '''

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

            if len(occ_au) >= 3:
                ori_prob_all_au = prob_all_au
                new_prob_all_au = prob_all_au

                new_prob_all_au = RadiateAUs(new_AU_cpt, occ_au, thresh=args.AUthresh)
                ori_prob_all_au = RadiateAUs(ori_AU_cpt, occ_au, thresh=args.AUthresh)

                ori_pos = np.where(ori_prob_all_au > args.AUthresh)[0]
                new_pos = np.where(new_prob_all_au > args.AUthresh)[0]
                ori_weight = np.array(ori_weight)
                new_weight = np.array(new_weight)

                ori_EMO_bonus = [1] * len(EMO)
                new_EMO_bonus = [1] * len(EMO)
                for i in range(ori_prob_all_au.shape[0]-2):
                    if i in ori_pos:
                        ori_prob_all_au[i] = ori_prob_all_au[i] / ori_prob_AU[i]
                    else:
                        ori_prob_all_au[i] = 1 / (1-ori_prob_AU[i])
                        ori_weight[i, :] = 1 - ori_weight[i, :]
                for i in range(new_prob_all_au.shape[0]-2):
                    if i in new_pos:
                        new_prob_all_au[i] = new_prob_all_au[i] / new_prob_AU[i]
                    else:
                        new_prob_all_au[i] = 1 / (1-new_prob_AU[i])
                        new_weight[i, :] = 1 - new_weight[i, :]
                
                ori_prob_all_au[-2] = 1 / ori_prob_AU[-2]
                ori_prob_all_au[-1] = 1 / ori_prob_AU[-1]
                new_prob_all_au[-2] = 1 / new_prob_AU[-2]
                new_prob_all_au[-1] = 1 / new_prob_AU[-1]
                if emo_label == 0:
                    ori_weight[-1, :] = 1 - ori_weight[-1, :]
                    new_weight[-1, :] = 1 - new_weight[-1, :]
                    ori_prob_all_au[-1] = 1 / (1-ori_prob_AU[-1])
                    ori_prob_all_au[-2] = 1 / ori_prob_AU[-2]
                    new_prob_all_au[-1] = 1 / (1-new_prob_AU[-1])
                    new_prob_all_au[-2] = 1 / new_prob_AU[-2]
                elif emo_label == 2 or emo_label == 4:
                    pass
                else:
                    ori_weight[-2, :] = 1 - ori_weight[-2, :]
                    ori_weight[-1, :] = 1 - ori_weight[-1, :]
                    new_weight[-2, :] = 1 - new_weight[-2, :]
                    new_weight[-1, :] = 1 - new_weight[-1, :]
                    ori_prob_all_au[-2] = 1 / (1-ori_prob_AU[-2])
                    ori_prob_all_au[-1] = 1 / (1-ori_prob_AU[-1])
                    new_prob_all_au[-2] = 1 / (1-new_prob_AU[-2])
                    new_prob_all_au[-1] = 1 / (1-new_prob_AU[-1])

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

    pass