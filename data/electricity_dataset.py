import os
import random
from torch.utils.data import Dataset
import numpy as np
import torch


# load data from files
class ElectricityDataset(Dataset):
    def __init__(self, dataroot, phase, shuffle=True, type=None):
        self.dataroot = dataroot
        self.phase = phase
        self.shuffle = shuffle
        self.sample_paths = []
        self.normal_sample_paths = []
        self.abnormal_sample_paths = []
        self.type = type
        print(phase)
        if self.phase == 'train':
            self.__load_sample_path_(self.dataroot + 'normal_train.txt')
            self.__load_sample_path_(self.dataroot + 'abnormal_train.txt')
            print('normal sample number: ', str(len(self.normal_sample_paths)))
            print('abnormal sample number: ', str(len(self.abnormal_sample_paths)))
            random.shuffle(self.sample_paths)
        else:
            self.__load_sample_path_(self.dataroot + 'normal_' + str(self.phase) + '.txt')
            self.__load_sample_path_(self.dataroot + 'abnormal_' + str(self.phase) + '.txt')
        print('all sample number:', str(self.__len__()))

    def __load_sample_path_(self, txt_path):
        print('load ', txt_path)
        for sample in open(txt_path):
            sample = sample.replace('\n', '')
            self.sample_paths.append(sample)
            if self.phase == 'train':
                if sample.find('normal') == 0:
                    self.normal_sample_paths.append(sample)
                elif sample.find('abnormal') == 0:
                    self.abnormal_sample_paths.append(sample)
                else:
                    assert (1 == 0), sample

    def __getitem__(self, index):
        if self.type is None:  # not pfsc training
            if self.shuffle is True:
                if index % 2 == 0:
                    index_new = random.randint(0, len(self.normal_sample_paths) - 1)
                    path = self.normal_sample_paths[index_new]
                else:
                    index_new = random.randint(0, len(self.abnormal_sample_paths) - 1)
                    path = self.abnormal_sample_paths[index_new]
            else:
                path = self.sample_paths[index]
            all_info = np.load(self.dataroot + path)
            elec = torch.from_numpy(all_info['elec']).type(torch.FloatTensor)
            flag = all_info['flag']
            flag = torch.from_numpy(np.array([flag])).type(torch.FloatTensor)
        elif self.type == 'abnormal':
            path = self.abnormal_sample_paths[index]
            all_info = np.load(self.dataroot + path)
            elec = torch.from_numpy(all_info['elec']).type(torch.FloatTensor)
            flag = all_info['flag']
            flag = torch.from_numpy(np.array([flag])).type(torch.FloatTensor)
        else:
            path = self.normal_sample_paths[index]
            all_info = np.load(self.dataroot + path)
            elec = torch.from_numpy(all_info['elec']).type(torch.FloatTensor)
            flag = all_info['flag']
            flag = torch.from_numpy(np.array([flag])).type(torch.FloatTensor)
        return {'elec': elec, 'flag': flag, 'path': path}

    def __len__(self):
        if self.type is None:
            return len(self.sample_paths)
        elif self.type == 'abnormal':
            return len(self.abnormal_sample_paths)
        else:
            return len(self.normal_sample_paths)


# load data from the given dataset
class DatasetForMetaCls(Dataset):
    def __init__(self, dataset, phase):
        self.elec_input = dataset["elec"]
        self.flag_input = dataset["flag"]
        self.path_input = dataset["path"]
        self.phase = phase

        normal_index = torch.where(self.flag_input.reshape(-1, 1).squeeze(1) == 0)[0]
        abnormal_index = torch.where(self.flag_input.reshape(-1, 1).squeeze(1) == 1)[0]

        self.elec_normal = self.elec_input[normal_index]
        self.elec_abnormal = self.elec_input[abnormal_index]
        self.flag_normal = self.flag_input[normal_index]
        self.flag_abnormal = self.flag_input[abnormal_index]
        self.path_normal, self.path_abnormal = [], []
        for n_idx in normal_index:
            self.path_normal.append(self.path_input[n_idx])
        for a_idx in abnormal_index:
            self.path_abnormal.append(self.path_input[a_idx])

    def __getitem__(self, index):
        if self.phase == 'train':
            if index % 2 == 0:
                index_new = random.randint(0, len(self.elec_normal) - 1)
                elec, flag, path = self.get_data(index_new, type='normal')
            else:
                index_new = random.randint(0, len(self.elec_abnormal) - 1)
                elec, flag, path = self.get_data(index_new, type='abnormal')
        else:
            # when testing, sample balance is not needed
            elec = self.elec_input[index].type(torch.FloatTensor)
            flag = self.flag_input[index].type(torch.FloatTensor)
            path = self.path_input[index]
        return {'elec': elec, 'flag': flag, 'path': path}

    def __len__(self):
        return len(self.elec_input)

    def get_data(self, index, type='normal'):
        elec, flag, path = None, None, None
        if type == 'abnormal':
            elec = self.elec_abnormal[index].type(torch.FloatTensor)
            flag = self.flag_abnormal[index].type(torch.FloatTensor)
            path = self.path_abnormal[index]
        elif type == 'normal':
            elec = self.elec_normal[index].type(torch.FloatTensor)
            flag = self.flag_normal[index].type(torch.FloatTensor)
            path = self.path_normal[index]
        return elec, flag, path