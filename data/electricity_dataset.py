import os
import random
from data.base_dataset import BaseDataset

import numpy as np
import torch


class ElectricityDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dataroot = opt.dataroot
        self.phase = opt.phase

        self.sample_paths = []
        self.normal_sample_paths = []
        self.abnormal_sample_paths = []

        print(opt.phase)
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
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """

        if self.phase == 'train':
            # when traning, we need to balance the ratios of normal sampels and abnormal samples
            if index % 2 == 0:
                index_new = random.randint(0, len(self.normal_sample_paths) - 1)
                path = self.normal_sample_paths[index_new]
            else:
                index_new = random.randint(0, len(self.abnormal_sample_paths) - 1)
                path = self.abnormal_sample_paths[index_new]
        else:
            # when testing, sample balance isnot needed
            path = self.sample_paths[index]

        all_info = np.load(self.dataroot + path)
        elec = torch.from_numpy(np.array(all_info['elec'])).type(torch.FloatTensor)
        flag = all_info['flag']
        flag = torch.from_numpy(np.array([flag])).type(torch.FloatTensor)
        '''
        print(type(elec))
        print(type(meta))
        print(elec.shape)
        print(meta.shape)
        print(flag)
        assert(1 == 0)
        '''

        return {'elec': elec, 'flag': flag, 'path': path}

    def __len__(self):
        """Return the total number of users in the dataset."""
        return len(self.sample_paths)