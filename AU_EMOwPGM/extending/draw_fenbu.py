import os,inspect
from queue import PriorityQueue
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
need_path = [current_dir, parent_dir, os.path.join(parent_dir,'models'), os.path.join(parent_dir,'extending')]
sys.path = need_path + sys.path
os.chdir(current_dir)

import datetime 
import pytz
import argparse
from easydict import EasyDict as edict

import numpy as np
import pandas as pd
import torch

from conf import ensure_dir, set_logger
from model_extend.utils_extend import *
from utils import *

def parser2dict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BP4D_all')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='draw_fenbu')
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

def main(conf):
    num_seen = 6
    num_unseen = 2
    pre_data_path = 'dataset'
    dataset_name = conf.dataset
    conf = get_config(conf)
    data_path = os.path.join(pre_data_path, dataset_name+'.pkl')
    data_info = torch.load(data_path, map_location='cpu')

    # 获取数据
    seen_train_inputAU = data_info['train_input_info']['seen_AU']
    seen_train_inputEMO = data_info['train_input_info']['seen_EMO']
    unseen_train_inputAU = data_info['train_input_info']['unseen_AU']
    unseen_train_inputEMO = data_info['train_input_info']['unseen_EMO']#+num_seen

    AU_data_all = torch.cat((seen_train_inputAU, unseen_train_inputAU))
    EMO_data_all = torch.cat((seen_train_inputEMO, unseen_train_inputEMO))
    data_all = (AU_data_all, EMO_data_all)

    AU_data_seen = seen_train_inputAU.clone()
    EMO_data_seen = seen_train_inputEMO.clone()
    data_seen = (AU_data_seen, EMO_data_seen)
    
    seen_trained_rules = data_info['seen_trained_rules']
    samplesAU1, samplesEMO1 = generate_seen_sample_v2(conf, seen_trained_rules)
    repeat_size = int((unseen_train_inputEMO.shape[0] // len(samplesEMO1))/num_unseen*num_seen)
    # repeat_size = int((unseen_train_inputEMO.shape[0]/num_unseen*num_seen)// len(samplesEMO1))
    samplesAU = samplesAU1.repeat(repeat_size, 1)
    samplesEMO = torch.concat(samplesEMO1 * repeat_size)
    AU_data_fake = torch.cat((samplesAU, unseen_train_inputAU))
    EMO_data_fake = torch.cat((samplesEMO, unseen_train_inputEMO))
    data_fake = (AU_data_fake, EMO_data_fake)

    conf.save_path = os.path.join(conf.save_path, dataset_name)
    ensure_dir(conf.save_path)
    checkpoint = {}
    checkpoint['data_all'] = data_all
    checkpoint['data_seen'] = data_seen
    checkpoint['data_fake'] = data_fake
    data_save_path = os.path.join(conf.save_path, 'data.pkl')
    torch.save(checkpoint, data_save_path)

    AU_record_all, EMO_record_all = do_fenbu(data_all)
    AU_record_seen, EMO_record_seen = do_fenbu(data_seen)
    AU_record_fake, EMO_record_fake = do_fenbu(data_fake)
    excel_save_path = os.path.join(conf.save_path, 'fenbu.xlsx')
    writer = pd.ExcelWriter(excel_save_path)		# 写入Excel文件
    AU_record_all.to_excel(writer, 'AU_all', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    AU_record_seen.to_excel(writer, 'AU_seen', float_format='%.5f')
    AU_record_fake.to_excel(writer, 'AU_fake', float_format='%.5f')
    EMO_record_all.to_excel(writer, 'EMO_all', float_format='%.5f')		
    EMO_record_seen.to_excel(writer, 'EMO_seen', float_format='%.5f')
    EMO_record_fake.to_excel(writer, 'EMO_fake', float_format='%.5f')
    writer.save()
    writer.close()

    a = 1

def do_fenbu(data):
    AU_data, EMO_data = data
    num_EMO = max(EMO_data) - 1
    start_EMO = min(EMO_data)
    num_AU = AU_data.shape[1]
    AU_record = []
    EMO_record = []

    for i in range(num_EMO):
        loc_idx = torch.where(EMO_data==i+start_EMO)[0]
        i_AU_data = AU_data[loc_idx, :]
        i_EMO_len = i_AU_data.shape[0]
        tmp_AU_record = torch.sum(i_AU_data, dim=0) / i_EMO_len
        AU_record.append(tmp_AU_record)
        EMO_record.append(i_EMO_len)
    AU_record = torch.stack(AU_record)
    AU_record = AU_record.detach().cpu().numpy()
    AU_record = pd.DataFrame(AU_record)
    EMO_record = np.array(EMO_record).reshape(1, -1)
    EMO_record = pd.DataFrame(EMO_record)
    return AU_record, EMO_record

if __name__=='__main__':
    conf = parser2dict()
    main(conf)
    a = 1