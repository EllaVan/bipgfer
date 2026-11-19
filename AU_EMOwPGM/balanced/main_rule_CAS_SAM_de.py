
'''
balanced 方法训练单个数据集
independent
'''
import os
import sys
current_dir = os.path.dirname(__file__) # 当前文件所属文件夹
parent_dir = os.path.dirname(current_dir) # 当前文件所属父文件夹
need_path = [current_dir, parent_dir]#, os.path.join(parent_dir,'models')]
sys.path = need_path + sys.path
os.chdir(current_dir)

import logging
import shutil
import argparse
from easydict import EasyDict as edict
import yaml
import datetime
import pytz

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from conf import ensure_dir, set_logger
from models.rule_model import learn_rules, test_rules
from utils import *

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--fold', type=int, default=1)
    # parser.add_argument('--dataset_order', type=str, default=['RAF-DB', 'AffectNet', 'BP4D', 'DISFA'])
    parser.add_argument('--dataset_order', type=str, default=['CASME', ])
    parser.add_argument('--outdir', type=str, default='save/rule/sub_de')
    parser.add_argument('-b','--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    
    parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--evaluate', action='store_true', help='evaluation mode')

    # --------------------settings for training-------------------
    parser.add_argument('--manualSeed', type=int, default=0)
    
    parser.add_argument('--lr_decay_idx', type=int, default=20000)
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-5)

    # --------------------settings for balanced-------------------
    parser.add_argument('--lr_relation', type=float, default=1e-5) #交叉CASME: 1e-5 SAMM 1e-4 不交叉SAMM 1e-5
    parser.add_argument('--isFocal_Loss', type=bool, default=True)
    parser.add_argument('--isClass_Weight', type=bool, default=False)
    parser.add_argument('--isClass_Weight_decay', type=bool, default=False)

    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

# 根据不同的数据集取不同的数据，同时更新args
def get_config(cfg):
    if cfg.dataset == 'CASME':
        with open('../config/CASME_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
    elif cfg.dataset == 'SAMM':
        with open('../config/SAMM_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    if cfg.fold == 0:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_independent
    else:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_dependent
    cfg.update(datasets_cfg)
    return cfg

def main(conf):
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        conf.dataset = dataset_name
        conf = get_config(conf)
        cur_outdir = os.path.join(conf.outdir, dataset_name) # 每个数据集的存储位置是conf.outdir/dataset
        ensure_dir(cur_outdir, 0)
        set_logger(conf)
        torch.cuda.empty_cache()
        # pre_path = os.path.join(pre_path1, dataset_name, pre_path2)

        info_source = conf.source_list # 单个数据集训练好的规则的类别来源（是label还是predict)，其epoch来源应与数据的epoch来源一致
        cur_path = conf.file_list # 数据的epoch来源

        # info_path = os.path.join(pre_path, cur_path) # 数据来源路径
        info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
        
        # 获取数据的AU和EMO情况，并设定输出
        with open(conf.dataset_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
        dataset_AU = pkl_file['AU']
        priori_AU = pkl_file['AU']
        AU_p_d = (priori_AU, dataset_AU)
        dataset_EMO = pkl_file['EMO']
        dataset_info = infolist(dataset_EMO, dataset_AU)

        input_rules = pkl_file['priori_rules']
        sub_file = pkl_file['sub']
        sub_list = sub_file['sub_list']
        num_sub = len(sub_list)
        num_EMO = len(dataset_EMO)
        val_confu_m = np.zeros((num_EMO, num_EMO))
        val_acc_list = []
        for sub_i, sub_n in enumerate(sub_list):
            # if sub_n != 25: # SAMM
            if sub_n!=8 and sub_n!=18 and sub_n!=21 and sub_n!=22: # CASMe
                val_inputAU = sub_file[sub_n]['labelsAU']
                val_inputEMO = sub_file[sub_n]['labelsEMO']
                val_inputAU = torch.from_numpy(np.stack(val_inputAU))
                val_inputEMO = torch.from_numpy(np.array(val_inputEMO))
                val_rules_input = (val_inputAU, val_inputEMO)

                train_inputAU = []
                train_inputEMO = []
                for sub_ai, sub_an in enumerate(sub_list):
                    if sub_an!=sub_n and sub_n!=8 and sub_n!=18 and sub_n!=21 and sub_n!=22:
                        train_inputAU = train_inputAU+(sub_file[sub_an]['labelsAU'])
                        train_inputEMO = train_inputEMO+(sub_file[sub_an]['labelsEMO'])
                train_inputAU = torch.from_numpy(np.stack(train_inputAU))
                train_inputEMO = torch.from_numpy(np.array(train_inputEMO))
                # train_inputAU, train_inputEMO = shuffle_input(train_inputAU, train_inputEMO)
                train_rules_input = (train_inputAU, train_inputEMO)
            
                # 训练与测试
                summary_writer = SummaryWriter(cur_outdir)
                output_rules, train_records, model = learn_rules(conf, train_rules_input, input_rules, AU_p_d, summary_writer)
                val_records = test_rules(conf, model, val_rules_input, output_rules, AU_p_d, summary_writer, val_confu_m)
                val_rules_loss, val_rules_acc, val_confu_m = val_records
                val_acc_list.append(val_rules_acc)

        val_info = {}
        val_info['val_acc_list'] = val_acc_list
        # del val_acc_list[17] # SAMM
        val_info['val_acc'] = np.mean(np.array(val_acc_list))
        val_info['val_confu_m'] = val_confu_m
        checkpoint = {}
        checkpoint['val_info'] = val_info
        checkpoint['output_rules'] = output_rules
        checkpoint['model'] = model
        ensure_dir(os.path.join(cur_outdir, info_source_path), 0)
        torch.save(checkpoint, os.path.join(cur_outdir, info_source_path, cur_path))

        infostr_rules = {'Dataset: {} val_rules_acc: {:.2f}'
                        .format(dataset_name, 100.* np.mean(np.array(val_acc_list)))}
        logging.info(infostr_rules)
        infostr_EMO = {'EMO Rules Val Acc-list:'}
        logging.info(infostr_EMO)
        val_confu_m2 = val_confu_m.copy()
        for i in range(val_confu_m.shape[0]):
            val_confu_m2[:, i] = val_confu_m2[:, i] / val_confu_m2[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(np.diag(val_confu_m2).tolist())
        logging.info(infostr_EMO)     
        end_flag = 1
    a = 1

if __name__=='__main__':
    setup_seed(0)
    conf = parser2dict()
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    # cur_day = str(cur_time).split('.')[0].replace(' ', '_')
    cur_time = str(cur_time).split('.')[0]
    cur_day = cur_time.split(' ')[0]
    cur_clock = cur_time.split(' ')[1]
    conf.outdir = os.path.join(conf.outdir, cur_day)

    global device
    conf.gpu = 1
    # device = torch.device('cuda:{}'.format(conf.gpu))
    device = torch.device('cuda:{}'.format(conf.gpu) if torch.cuda.is_available() else "cpu")
    conf.device = device
    # torch.cuda.set_device(conf.gpu)
    # ensure_dir(conf.outdir, 0)
    # set_logger(conf)
    # shutil.copyfile("../models/rule_model.py", os.path.join(conf.outdir,'rule_model.py'))
    # shutil.copyfile("./main_rule.py", os.path.join(conf.outdir,'main_rule.py'))
    main(conf)
    a = 1