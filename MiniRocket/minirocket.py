# from fastai.torch_core import default_device
# from tsai.metrics import F1_multi
# from util.electricity_evaluation import evaluation
from tsai.all import *
from fastai.metrics import accuracy, F1Score
from fastai.callback.tracker import ReduceLROnPlateau
import os
import time
import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as data
from abc import ABC, abstractmethod
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix

# The compared method MiniRocket is from the Pytorch implementation from an open-source deep learning package tsai
# https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb


class BaseDataset(data.Dataset, ABC):
    def __init__(self, dataroot, phase):
        self.root = dataroot
        self.phase = phase
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    @abstractmethod
    def __len__(self):
        return 0
    @abstractmethod
    def __getitem__(self, index):
        pass


class ElectricityDataset(BaseDataset):
    def __init__(self, dataroot, phase):
        BaseDataset.__init__(self, dataroot, phase)
        self.dataroot = dataroot
        self.phase = phase
        self.sample_paths = []
        self.normal_sample_paths = []
        self.abnormal_sample_paths = []
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
        if self.phase == 'train':
            if index % 2 == 0:
                index_new = random.randint(0, len(self.normal_sample_paths) - 1)
                path = self.normal_sample_paths[index_new]
            else:
                index_new = random.randint(0, len(self.abnormal_sample_paths) - 1)
                path = self.abnormal_sample_paths[index_new]
        else:
            path = self.sample_paths[index]
        all_info = np.load(self.dataroot + path)
        elec = torch.from_numpy(all_info['elec'])
        flag = all_info['flag']
        flag = torch.from_numpy(np.array([flag]).astype(np.float32))
        return {'elec': elec, 'flag': flag}

    def __len__(self):
        return len(self.sample_paths)

def evaluation(result_dataframe, best_threshold=0, trad=False):
    pred = result_dataframe['pred'].values
    flag = result_dataframe['flag'].values.astype(np.int)
    print("in eval, pred flag shape:", pred.shape, flag.shape)
    assert (len(pred) == len(flag))
    best_f1_score = 0
    if trad:
        best_f1_threshold = best_threshold
        preds=np.where(pred>best_threshold, 1, 0) # best_f1_precision = sum(pred == flag)*1.0 / len(pred)
        conf_matrix = confusion_matrix(flag, preds)
        try:
            tp = conf_matrix[1][1]
        except:
            tp = 0
        try:
            tn = conf_matrix[0][0]
        except:
            t = 0
        try:
            fp = conf_matrix[0][1]
        except:
            fp = 0
        try:
            fn = conf_matrix[1][0]
        except:
            fn = 0
        if (tp != 0 or fp != 0):
            best_f1_precision = tp / (tp + fp)
        else:
            best_f1_precision = 0.0
        if (tp != 0 or fn != 0):
            best_f1_recall = tp / (tp + fn)
        else:
            best_f1_recall = 0.0
        if (best_f1_precision != 0.0 or best_f1_recall != 0):
            best_f1_score = 2 * best_f1_precision * best_f1_recall / (best_f1_precision + best_f1_recall)
    else:
        precisions, recalls, thresholds = precision_recall_curve(flag, pred)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.where(f1_scores == best_f1_score)[0]
        best_f1_threshold   = thresholds[best_f1_score_index]
        best_f1_precision   = precisions[best_f1_score_index]
        best_f1_recall      = recalls[best_f1_score_index]
    fpr, tpr, th = roc_curve(flag, pred, pos_label=1)
    elec_auc = auc(fpr, tpr)
    MAP = mean_average_precision(pred, flag, len(flag))
    return best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP

def mean_average_precision(pred, flag, topk):
    # pred has been sorted.
    AP = 0.0
    pos_num = 0.0
    if topk > len(pred):
        print("topk > all")
        return AP
    for i in range(topk):
        if flag[i] == 1:
            pos_num = pos_num + 1.0
            AP = AP + pos_num / (i + 1)
    assert(pos_num > 0.0), 'pos_num is 0'
    AP = AP / pos_num
    return AP

def load_datasets(dataroot):
    dataset1 = ElectricityDataset(dataroot, 'train')
    dataset_train = torch.utils.data.DataLoader(dataset1, batch_size=len(dataset1), shuffle=True, num_workers=4)
    dataset_train_iter = iter(dataset_train)
    data_train = dataset_train_iter.__next__()
    elec_data = data_train['elec']
    flag_data = data_train['flag']
    dataset2 = ElectricityDataset(dataroot, 'test')
    dataset_test = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2), shuffle=False, num_workers=4)
    dataset_test_iter = iter(dataset_test)
    data_test = dataset_test_iter.__next__()
    elec_data_test = data_test['elec']
    flag_data_test = data_test['flag']
    X = torch.cat([elec_data.view(-1, 2, 148*7), elec_data_test.view(-1, 2, 148*7)], dim=0)
    y = torch.cat([flag_data.squeeze(-1), flag_data_test.squeeze(-1)], dim=0)
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    splits = ([i for i in range(elec_data.shape[0])],
              [j for j in range(elec_data.shape[0], elec_data.shape[0] + elec_data_test.shape[0])])
    print('training: {:.2f} positive ratio with {}, shape{}'.format(sum(y[splits[0]]) * 1.0 / len(y[splits[0]]),
                                                                    len(y[splits[0]]), X[splits[0]].shape))
    print('testing: {:.2f} positive ratio with {}, shape{}'.format(sum(y[splits[1]]) * 1.0 / len(y[splits[1]]),
                                                                   len(y[splits[1]]), X[splits[1]].shape))
    return X, y, splits

def generate_features_512(X, splits):
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]) #.to(default_device())
    X_train = X[splits[0]]
    mrf.fit(X_train)
    # X_feat = get_minirocket_features(X, mrf, chunksize=1024, to_np=True)
    feat_generated = get_minirocket_features(X, mrf, chunksize=512, to_np=True)
    print("X_feat:", feat_generated.shape)
    num_params = 0
    for param in mrf.named_parameters():
        print(param[0], ":", param[1].numel())
        num_params += param[1].numel()
    print('Total number of parameters for [mrf]: %.3f M' % (num_params / 1e6))
    PATH = Path("../results/minirocket/MRF.pt")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), PATH)
    return feat_generated

def generate_features(X, splits):
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]) #.to(default_device())
    X_train = X[splits[0]]
    mrf.fit(X_train)
    feat_generated = get_minirocket_features(X, mrf, chunksize=1024, to_np=True)
    print("X_feat:", feat_generated.shape)
    num_params = 0
    for param in mrf.named_parameters():
        print(param[0], ":", param[1].numel())
        num_params += param[1].numel()
    print('Total number of parameters for [mrf]: %.3f M' % (num_params / 1e6))
    PATH = Path("../results/minirocket/MRF.pt")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), PATH)
    return feat_generated

def create_model(X_feat, y, splits, batch_tfms=None, batch_size=64):
    tfms = [None, TSClassification()]
    if batch_tfms == 'empty':
        batch_tfms = TSStandardize()
    elif batch_tfms == 'var':
        batch_tfms = TSStandardize(by_var=True)
    elif batch_tfms == 'sample':
        batch_tfms = TSStandardize(by_sample=True)
    else:
        batch_tfms = None
    dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=batch_size)  # TSDataLoaders
    print("dls.d, c_in, out, seq_len:", dls.d, dls.vars, dls.c, dls.len)
    # model is a linear classifier Head
    model = build_ts_model(MiniRocketHead, dls=dls)
    num_params = 0
    for param in model.named_parameters():
        print(param[0], ":", param[1].numel())
        num_params += param[1].numel()
    print('Total number of parameters for [model]: %.3f M' % (num_params / 1e6))
    return dls, model

def learning(dls, model, epoch, lr, metrics="acc"):
    if metrics == 'f1':
        learn = Learner(dls, model, metrics=F1_multi)
    elif metrics == 'mse':
        learn = Learner(dls, model, metrics=mse)
    else:
        learn = Learner(dls, model, metrics=accuracy)
    best_lr = float(learn.lr_find().valley)
    print("best lr:", best_lr)
    timer.start()
    learn.fit(n_epoch=epoch, lr=best_lr, cbs=ReduceLROnPlateau(factor=0.5, min_lr=1e-8, patience=5))
    timer.stop()
    PATH = Path('../results/minirocket/MRL.pkl')
    PATH.parent.mkdir(parents=True, exist_ok=True)
    learn.export(PATH)
    return

def inference(X, splits):
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]) #.to(default_device())
    PATH = Path("../results/minirocket/MRF.pt")
    mrf.load_state_dict(torch.load(PATH))
    new_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=1024, to_np=True)
    PATH = Path('../results/minirocket/MRL.pkl')
    learn_for_infer = load_learner(PATH, cpu=False)
    probas, _, preds = learn_for_infer.get_X_preds(new_feat)
    probas = np.array(probas).astype(np.float)
    preds = preds.astype(np.float)
    return probas, preds

def inference_512(X, splits):
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]) #.to(default_device())
    PATH = Path("../results/minirocket/MRF.pt")
    mrf.load_state_dict(torch.load(PATH))
    # new_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=1024, to_np=True)
    new_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=512, to_np=True)
    PATH = Path('../results/minirocket/MRL.pkl')
    learn_for_infer = load_learner(PATH, cpu=False)
    probas, _, preds = learn_for_infer.get_X_preds(new_feat)
    probas = np.array(probas).astype(np.float)
    preds = preds.astype(np.float)
    return probas, preds

def eval(preds, all_flag, trad=True):
    dataframe = pd.DataFrame({'pred': np.array(preds), 'flag': all_flag})
    dataframe = dataframe.sort_values(by=['pred'], ascending=False)
    dataframe.to_csv("../results/minirocket/minirocket_result.csv", sep=',')
    best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP = evaluation(dataframe, trad=trad)
    print("best_f1_threshold:", best_f1_threshold)
    print("best_f1_score:", best_f1_score)
    print("best_f1_precision:", best_f1_precision)
    print("best_f1_recall:", best_f1_recall)
    print("elec_auc:", elec_auc)
    print("MAP:", MAP)
    return

if __name__ == '__main__':
    dataroot = "../datasets/electricity/"
    results_save_dir = "../results/minirocket"
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)

    X, y, splits = load_datasets(dataroot)
    train_start_time = time.time()

    X_feat_512 = generate_features_512(X, splits)

    dls_512, model_512 = create_model(X_feat_512, y, splits, batch_tfms='empty')

    # learn = Learner(dls_512, model_512, metrics=accuracy)
    # learn.lr_find()

    learning(dls_512, model_512, epoch=20, lr=4*1e-5)
    print("training time:%.2f" % (time.time() - train_start_time))

    test_start_time = time.time()
    probas, preds = inference_512(X, splits)
    print("test time:%.2f" % (time.time() - test_start_time))

    eval(probas[:, 1], y[splits[1]], trad=False)





