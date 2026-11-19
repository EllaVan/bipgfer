import torch
import pickle as pkl

path1 = 'E:/DeskTop/BiPG_FER/codes/0114/AU_EMOwPGM/dataset/CASME/CASME_subject_independent.pkl'
path2 = 'E:/DeskTop/BiPG_FER/codes/0114/AU_EMOwPGM/balanced/save/rule/CK+/predsAU_labelsEMO/epoch1_model_fold0.pth'

with open(path1, 'rb') as fo:
    pkl_file = pkl.load(fo)

# info = torch.load(path2, map_location='cpu')
# output_rules = info['output_rules']
# val_confu_m = info['val_info']['val_confu_m']
# for i in range(val_confu_m.shape[0]):
#     val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0) * 100.00
a = 1