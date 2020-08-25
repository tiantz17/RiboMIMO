import time
import os
import time
import math
import json
import pickle
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import *


class RiboMIMOModel(object):
    """
    Predicting ribosome density from CDS
    """
    def __init__(self, args):
        super(RiboMIMOModel, self).__init__()
        """ parameters """
        assert args.RNN_model in ["RiboMIMO"]
        self.RNN_model = args.RNN_model
        self.RNN_output_dims = args.RNN_hidden_dims
        self.RNN_depth = args.RNN_depth
        self.add_nt = args.nt
        self.add_aa = args.aa
        self.threshold = args.threshold
        self.alpha = args.alpha
        self.sim = args.sim
        self.use_median = args.use_median
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.seed = args.seed
        self.info = args.info
        self.use_cuda = args.gpu != "-1"
        self.dataset = args.dataset


        """ constant parameters """
        self.input_dims = 64 # codon onehot encoding
        encode = "_codon"
        if self.add_nt:
            self.input_dims += 4 * 3
            encode += "_nt"
        if self.add_aa:
            self.input_dims += 21
            encode += "_aa"


        self.num_repeat = args.num_repeat # n times k fold cross-validation
        self.num_fold = args.num_fold # n times k fold cross-validation

        """ local directory """
        self.path = "../"
    
        file_folder = "results/{}/RiboMIMO{}_{}_{}_{}_{}_sim_{}_bs_{}_cv_{}x{}_{}"
        file_folder = file_folder.format(self.dataset, encode, self.threshold, self.alpha, 
                                         self.RNN_output_dims, self.RNN_depth,
                                         self.sim, self.batch_size,
                                         self.num_repeat, self.num_fold, self.info)
        file_folder += time.strftime("_%Y%m%d_%H%M%S/", time.localtime())
        self.save_path = self.path + file_folder
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.define_logging()
        logging.info("Local folder created: {}".format(self.save_path))

        """ save hyperparameters """
        self.save_hyperparameter()

        """ load data """
        self.load_data()

    def define_logging(self):
        # Create a logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            filename=self.save_path + "logging.log",
            filemode='w')
        # Define a Handler and set a format which output to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def save_hyperparameter(self):
        params = {'path': self.path,
                  'RNN_model': self.RNN_model,
                  'RNN_output_dims': self.RNN_output_dims,
                  'RNN_depth': self.RNN_depth,
                  'add_nt': self.add_nt,
                  'add_aa': self.add_aa, 
                  'threshold': self.threshold,
                  'alpha': self.alpha,
                  'batch_size': self.batch_size,
                  'max_epoch': self.max_epoch,
                  'early_stop': self.early_stop,
                  'learning_rate': self.learning_rate,
                  'weight_decay': self.weight_decay,
                  'seed': self.seed,
                  'info': self.info,
                  'use_cuda': self.use_cuda,
                  }

        json.dump(params, open(self.save_path + "config", "w+"))

    def load_data(self):
        logging.info("Loading data...")
        self.data = DataLoader(self.path, self.dataset, self.use_median)
        logging.info("{} genes used for training and testing".format(len(self.data.list_gene)))

    def split_data(self, seed):
        logging.info("Spliting data using threshold {} with seed {}".format(self.sim, seed))
        self.data.split_data(seed, self.num_fold, self.sim)

    @staticmethod
    def padding(data):
        num_data = len(data)
        length = max([len(i) for i in data])
        data_pad = []
        mask = []
        for i in range(num_data):
            data_pad.append(list(data[i]) + [np.array(data[i][0]) * 0] * (length - len(data[i])))
            temp = np.zeros(length)
            temp[:len(data[i])] = 1
            mask.append(temp)
        data_pad = np.array(data_pad)
        mask = np.array(mask)
        return data_pad, mask

    def get_data_batch(self, batch_index):
        index = self.data.length[batch_index].argsort()[::-1]
        batch_index = batch_index[index]
        x_input = self.data.x_input[batch_index]
        length = self.data.length[batch_index]
        density = self.data.density[batch_index]
        level = self.data.density_level[batch_index]
        # zero padding
        x_input, _ = self.padding(x_input)
        density, _ = self.padding(density)
        level, _ = self.padding(level)
        mask_density = np.array(density > 0, dtype=float)

        x_input = torch.Tensor(x_input)
        length = torch.Tensor(length)
        density = torch.Tensor(density)
        level = torch.LongTensor(level)
        mask_density = torch.BoolTensor(mask_density)
        if self.use_cuda:
            x_input = x_input.cuda()
            length = length.cuda()
            density = density.cuda()
            level = level.cuda()
            mask_density = mask_density.cuda()

        return x_input, length, density, mask_density, level, batch_index

    def load_model(self):
        logging.info("Loading model...")
        self.model = RiboMIMO(self.input_dims,
                              self.RNN_output_dims,
                              self.RNN_depth,
                              num_class=len(self.threshold.split(","))+1)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        self.MSELoss = nn.MSELoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.95)
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.apply(weights_init)

    def train(self):
        np.random.seed(self.seed)
        seeds = np.random.randint(0, 1024, self.num_repeat)
        logging.info("{} times {} fold cross-validation".format(self.num_repeat, self.num_fold))
        scores = []
        for repeat in range(self.num_repeat):
            self.split_data(seeds[repeat])
            for fold in range(self.num_fold):
                logging.info("{}/{} repeat {}/{} fold".format(repeat+1, self.num_repeat, fold+1, self.num_fold))
                self.load_model()
                score = self.train_one_fold(repeat, fold)
                scores.append(score)
        scores = np.array(scores)
        scores_mean = np.mean(scores, 0)
        scores_std = np.std(scores, 0)
        # print score
        logging.info("Done {} repeat {} fold cross-validation".format(self.num_repeat, self.num_fold))
        printstring = "\033[91mRibosome, r2 {:.6f} {:.6f}, mse {:.6f} {:.6f}, corr {:.6f} {:.6f}\033[0m"
        logging.info(printstring.format(scores_mean[0], scores_std[0], scores_mean[1], scores_std[1], scores_mean[2], scores_std[2]))

    def get_thresh(self):
        list_density = []
        for gene in self.index_train:
            list_density.extend(self.data.dict_density[gene])
        list_density = np.array(list_density)
        list_density = list_density[list_density>0]
        list_density = np.log10(list_density)
        avg = np.mean(list_density)
        std = np.std(list_density)
        self.threshs = []
        thresholds = np.array(self.threshold.split(","), dtype=float)
        thresholds = np.sort(thresholds)
        for i in thresholds:
            self.threshs.append(avg + float(i) * std)

    def train_one_fold(self, repeat, fold):
        """ train the model for one fold"""
        """ get train valid test set """
        self.index_test = self.data.list_fold[fold]
        self.index_train = self.data.list_fold[:fold] + self.data.list_fold[fold+1:]
        self.index_train = np.concatenate(self.index_train)
        num_valid = len(self.index_train) // 10
        self.index_valid = self.index_train[:num_valid]
        self.index_train = self.index_train[num_valid:]
        self.num_train = len(self.index_train)
        self.get_thresh()

        logging.info("training data size: {}".format(len(self.index_train)))
        logging.info("validation data size: {}".format(len(self.index_valid)))
        logging.info("test data size: {}".format(len(self.index_test)))

        best_score = -np.inf
        best_score_list = [0] * 6
        no_improve = 0
        for epoch in range(self.max_epoch):
            no_improve += 1
            logging.info("Epoch {}".format(epoch+1))
            # train
            self.train_one_epoch(repeat, fold, epoch)
            self.scheduler.step()
            # evaluate
            for evalset in ["Valid", "Test"]:
                r2, mse, corr, auc, aupr, loss_cls = self.evaluate(evalset)
                printstring = "\033[91mFold {}x{}/{}x{}, {}, r2 {:.6f}, mse {:.6f}, corr {:.6f}, auc {:.6f}, aupr {:.6f}, loss {:.6f}\033[0m"
                logging.info(printstring.format(repeat+1, fold+1, self.num_repeat, self.num_fold, evalset, r2, mse, corr, auc, aupr, loss_cls))
                if evalset == "Valid":
                    if best_score < r2:
                        best_score = r2
                        self.save(repeat, fold)
                        no_improve = 0
                if evalset == "Test":
                    if no_improve == 0:
                        best_score_list = [r2, mse, corr]
            if no_improve >= self.early_stop:
                logging.info("Early stopped after {} epochs".format(epoch+1))
                break
        return best_score_list

    def train_one_epoch(self, repeat, fold, epoch):
        self.data.get_data_pack(self.index_train, self.add_nt, self.add_aa)
        self.data.get_data_pack_cls(self.index_train, self.threshs)
        index_shuffle = np.array(range(self.num_train))
        num_batch = int(np.ceil(self.num_train / self.batch_size))
        np.random.shuffle(index_shuffle)
        running_score = [[] for _ in range(7)]
        loss_cls = 0.0
        for batch in range(num_batch):
            index = index_shuffle[batch*self.batch_size:min((batch+1)*self.batch_size,self.num_train)]
            x_input, length, density, mask_density, level, batch_index = self.get_data_batch(index)
            self.optimizer.zero_grad()
            pred_reg, pred_cls, loss_cls = self.model(x_input, length)
            pred_reg = pred_reg.masked_select(mask_density)
            pred_cls = pred_cls.masked_select(mask_density.unsqueeze(2).repeat(1, 1, self.model.num_state)).view(-1, self.model.num_state)
            density = density.masked_select(mask_density)
            level = level.masked_select(mask_density)
            loss_reg = self.MSELoss(pred_reg, density)
            loss_cls = self.CrossEntropyLoss(pred_cls, level)
            loss = loss_reg + loss_cls * self.alpha
            loss.backward()
            self.optimizer.step()
            r2, mse, corr = self.model.score_reg(pred_reg.cpu().data.numpy(), density.cpu().data.numpy())
            auroc, aupr = self.model.score_cls(pred_cls.cpu().data.numpy(), level.cpu().data.numpy())
            running_score[0].append(loss_reg.cpu().data.numpy())
            running_score[1].append(r2)
            running_score[2].append(mse)
            running_score[3].append(corr)
            running_score[4].append(auroc)
            running_score[5].append(aupr)
            running_score[6].append(loss_cls.cpu().data.numpy())
            if (batch+1) % 10 == 0:
                printstring = "Fold {}x{}/{}x{}, Epoch {}, Batch {}/{}, loss {:.6f}, r2 {:.6f}, mse {:.6f}, corr {:.6f}, auc {:.6f}, aupr {:.6f}, loss {:.6f}"
                logging.info(printstring.format(repeat+1, fold+1, self.num_repeat, self.num_fold,
                                                epoch+1, batch+1, num_batch,
                                                np.nanmean(running_score[0]),
                                                np.nanmean(running_score[1]),
                                                np.nanmean(running_score[2]),
                                                np.nanmean(running_score[3]),
                                                np.nanmean(running_score[4]),
                                                np.nanmean(running_score[5]),
                                                np.nanmean(running_score[6])))
                running_score = [[] for _ in range(7)]

    def evaluate(self, evalset):
        if evalset == "Valid":
            index_eval = self.index_valid
        elif evalset == "Test":
            index_eval = self.index_test
        elif evalset == "Train":
            index_eval = self.index_train
        else:
            raise ValueError
        self.data.get_data_pack(index_eval, self.add_nt, self.add_aa)
        self.data.get_data_pack_cls(index_eval, self.threshs)
        num_batch = int(np.ceil(len(index_eval) / float(self.batch_size)))
        list_density = []
        list_pred_reg = []
        list_level = []
        list_pred_cls = []
        list_loss_cls = []
        loss_cls = 0.0
        with torch.no_grad():
            for batch in range(num_batch):
                index = np.array(range(batch*self.batch_size, min((batch+1)*self.batch_size,len(index_eval))))
                x_input, length, density, mask_density, level, batch_index = self.get_data_batch(index)
                pred_reg, pred_cls, loss_cls = self.model(x_input, length)

                pred_reg = pred_reg.masked_select(mask_density)
                pred_cls = pred_cls.masked_select(mask_density.unsqueeze(2).repeat(1, 1, self.model.num_state)).view(-1, self.model.num_state)
                density = density.masked_select(mask_density)
                level = level.masked_select(mask_density)

                list_density.extend(density.cpu().data.numpy())
                list_pred_reg.extend(pred_reg.cpu().data.numpy())
                list_level.extend(level.cpu().data.numpy())
                list_pred_cls.extend(pred_cls.cpu().data.numpy())
                loss_cls = self.CrossEntropyLoss(pred_cls, level).cpu().data.numpy()
                list_loss_cls.append(loss_cls)
        r2, mse, corr = self.model.score_reg(list_pred_reg, list_density)
        auc, aupr = self.model.score_cls(np.array(list_pred_cls), np.array(list_level))
        return r2, mse, corr, auc, aupr, np.mean(list_loss_cls)

    def save(self, repeat, fold):
        filename = self.save_path + "best_model_{}_{}.pth".format(repeat, fold)
        torch.save(self.model.state_dict(), filename)
        logging.info("Best model saved")


def main():
    parser = argparse.ArgumentParser()
    # define environment
    parser.add_argument("--gpu", default="0", help="which GPU to use", type=str)
    parser.add_argument("--seed", default=1234, help="random seed", type=int)
    parser.add_argument("--info", default="train", help="output folder special marker", type=str)
    # define data
    parser.add_argument("--dataset", default="Subtelny14", help="dataset", type=str)
    parser.add_argument("--sim", default=-1, help="similarity between folds", type=float)
    parser.add_argument("--use_median", default=False, action="store_true", help="using median")
    # define model
    parser.add_argument("--nt", default=False, action="store_true", help="nt encoding")
    parser.add_argument("--aa", default=False, action="store_true", help="aa encoding")    

    parser.add_argument("--RNN_model", default="RiboMIMO", help="RNN models", type=str)
    parser.add_argument("--RNN_hidden_dims", default=256, help="RNN_hidden_dims", type=int)
    parser.add_argument("--RNN_depth", default=2, help="RNN_depth", type=int)
    parser.add_argument("--threshold", default="0,2", help="threshold of classification", type=str)
    parser.add_argument("--alpha", default=3, help="alpha", type=float)
    # define training
    parser.add_argument("--num_repeat", default=1, help="number of cv repeats", type=int)
    parser.add_argument("--num_fold", default=10, help="number of cv folds", type=int)
    parser.add_argument("--batch_size", default=16, help="batch size", type=int)
    parser.add_argument("--max_epoch", default=100, help="max epoch", type=int)
    parser.add_argument("--early_stop", default=5, help="epochs for early stop", type=int)
    parser.add_argument("--learning_rate", default=1e-3, help="learning rate", type=float)
    parser.add_argument("--weight_decay", default=1e-6, help="weight decay", type=float)
    args = parser.parse_args()

    """ set gpu """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """ claim class instance """
    RiboMIMONet = RiboMIMOModel(args)

    """ Train """
    RiboMIMONet.train()




if __name__ == "__main__":
    main()
