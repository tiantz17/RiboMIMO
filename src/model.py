import os
import time
import pickle
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score

import utils

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        #print 'initial'
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0)

       
class RiboMIMO(nn.Module):
    """
    RNN model for predicting ribosome density using sequence data
    input:
            input_dims:    embedding size of input
            output_dims:   number of output channels
            RNN_depth:     number of RNN layers
    """
    def __init__(self, input_dims, output_dims, RNN_depth, num_class=3):
        super(RiboMIMO, self).__init__()
        self.rnn = nn.GRU(input_dims, output_dims, RNN_depth,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(output_dims*2, output_dims),
            nn.ReLU()
            )
        self.fc_reg = nn.Linear(output_dims, 1)
        self.num_state = num_class
        self.fc_cla = nn.Linear(output_dims, self.num_state)

    def score_reg(self, pred, label):
        r2 = r2_score(label, pred)
        mse = mean_squared_error(label, pred)
        corr = pearsonr(label, pred)[0]
        return r2, mse, corr

    def score_cls(self, pred, label):
        label = np.eye(self.num_state)[label]
        auroc, aupr = 0.0, 0.0
        try:
            auroc = roc_auc_score(label, pred)
            aupr = average_precision_score(label, pred)
        except:
            pass
        return auroc, aupr

    def get_rnn_feature(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        return x

    def forward(self, x, length):
        x = self.get_rnn_feature(x, length)
        x = self.fc(x)
        y = self.fc_reg(x).squeeze(2)
        z = self.fc_cla(x)
        return y, z, None
    
    def predict(self, x, length):
        x = self.get_rnn_feature(x, length)
        x = self.fc(x)
        y = self.fc_reg(x).squeeze(2)
        return y


class DataLoader(object):
    """ Data loader for riboseq data"""
    def __init__(self, path, dataset, use_median=False):
        self.path = path + "data/"
        self.dataset = dataset
        self.filename = dataset + ".txt"
        if not os.path.exists(self.path + self.filename):
            raise NotImplementedError("No data found for dataset {}".format(dataset))
        self.aa_table = utils.aa_table
        self.codon2aa = utils.codon2aa
        self.aa2codon = {}
        for codon in self.codon2aa.keys():
            if self.codon2aa[codon] not in self.aa2codon:
                self.aa2codon[self.codon2aa[codon]] = {codon}
            else:
                self.aa2codon[self.codon2aa[codon]].add(codon)
        # construct onehot encoding
        self.load_onehot()

        # load data from fasta
        self.load_data(use_median)

    def load_onehot(self):
        nts = "ATCG"
        codon2idx = {}
        for nt1 in nts:
            for nt2 in nts:
                for nt3 in nts:
                    codon2idx[nt1+nt2+nt3] = len(codon2idx)
        self.onehot_nt = {nts[i]:np.eye(4)[i] for i in range(len(nts))}
        self.onehot_codon = {codon:np.eye(64)[codon2idx[codon]] for codon in codon2idx}
        self.onehot_aa = {self.aa_table[i,2]:np.eye(21)[i] for i in range(len(self.aa_table))}
        self.codon2idx = codon2idx
        self.nt2idx = {nts[i]:i for i in range(4)}
        self.aa2idx = {self.aa_table[i,2]:i for i in range(len(self.aa_table))}

    def load_data(self, use_median=False):
        with open(self.path + self.filename, "r") as f:
            data = f.readlines()
        assert len(data)%3 == 0
        list_name = []
        list_seq = []
        list_density = []
        list_criteria = []
        list_avg = []
        for i in range(len(data)//3):
            name = data[3*i+0].split('>')[1].split()[0]
            seq = data[3*i+1].split()
            count = [float(e) for e in data[3*i+2].split()]
            if use_median:
                avg = np.median(np.array(count)[np.array(count)>0.5])
            else:
                avg = np.mean(np.array(count)[np.array(count)>0.5])
                
            density = (np.array(count)>0.5) * np.array(count) / avg
            # criteria containing ribosome density AND coverage percentage
            list_criteria.append([np.sum(count), np.mean(count), np.sum(np.array(count)>0.5), np.mean(np.array(count)>0.5)])
            list_name.append(name)
            list_seq.append(seq)
            list_density.append(density)
            list_avg.append(avg)

        list_name = np.array(list_name)
        list_seq = np.array(list_seq)
        list_density = np.array(list_density)
        list_avg = np.array(list_avg)
        list_criteria = np.array(list_criteria)
        self.list_gene_all = list_name
        # Coverage
        index = (list_criteria[:, 3]>0.6)
        self.list_gene_coverage_6 = list_name[index]
        index = (list_criteria[:, 3]>0.7)
        self.list_gene_coverage_7 = list_name[index]
        index = (list_criteria[:, 3]>0.8)
        self.list_gene_coverage_8 = list_name[index]
        index = (list_criteria[:, 3]>0.9)
        self.list_gene_coverage_9 = list_name[index]

        self.dict_seq = {}
        self.dict_seq_nt = {}
        self.dict_seq_aa = {}
        self.dict_seq_codon = {}
        self.dict_density = {}
        self.dict_avg = {}
        index = []
        for i in range(len(self.list_gene_all)):
            codons = list_seq[i]
            nts = "".join(codons)
            if "N" in nts:
                continue
            aas = [self.codon2aa[codon] for codon in codons]
            index.append(i)
            self.dict_seq[self.list_gene_all[i]] = nts
            self.dict_seq_nt[self.list_gene_all[i]] = np.array([self.onehot_nt[nt] for nt in nts])
            self.dict_seq_codon[self.list_gene_all[i]] = np.array([self.onehot_codon[codon] for codon in codons])
            self.dict_seq_aa[self.list_gene_all[i]] = np.array([self.onehot_aa[aa] for aa in aas])
            self.dict_density[self.list_gene_all[i]] = list_density[i]
            self.dict_avg[self.list_gene_all[i]] = list_avg[i]
        self.list_gene_all = self.list_gene_all[index]
        self.list_gene_coverage_6 = np.intersect1d(self.list_gene_coverage_6, self.list_gene_all)
        self.list_gene_coverage_7 = np.intersect1d(self.list_gene_coverage_7, self.list_gene_all)
        self.list_gene_coverage_8 = np.intersect1d(self.list_gene_coverage_8, self.list_gene_all)
        self.list_gene_coverage_9 = np.intersect1d(self.list_gene_coverage_9, self.list_gene_all)
        
        # default
        self.list_gene = self.list_gene_coverage_6
        logging.info("{} genes left after filtering in {}".format(len(self.list_gene_all), self.dataset))

    def split_data(self, seed, num_fold, sim):
        # only top 500 genes are used?
        # all data filtered should be useful
        if sim == -1:
            logging.info("Using random split")
            self.split_data_random(seed, num_fold)
            return
        try:
            ecoli_sim_mat = np.load(self.path + "gene_sim_mat.npy")
            list_name = np.load(self.path + "gene_name.npy")
            name_to_index = {list_name[i]:i for i in range(len(list_name))}
            index = np.array([name_to_index[gene] for gene in self.list_gene])
            ecoli_sim_mat = ecoli_sim_mat[index][:,index]
            name_to_index = {self.list_gene[i]:i for i in range(len(self.list_gene))}
        except:
            logging.info("Using random split")
            self.split_data_random(seed, num_fold)
            return
        self.list_fold = [[] for _ in range(num_fold)]
        list_available = list(self.list_gene).copy()
        np.random.seed(seed)
        np.random.shuffle(list_available)
        while len(list_available)>0:
            kernel = list_available.pop()
            list_queue = [kernel]
            j = 0
            while j<len(list_queue):
                list_new = list(self.list_gene[ecoli_sim_mat[name_to_index[list_queue[j]]] > sim])
                for i in list_new:
                    if i in list_available:
                        list_queue.append(i)
                        list_available.remove(i)
                j += 1
            idx = np.argmin([len(i) for i in self.list_fold])
            self.list_fold[idx].extend(list_queue)   

    def split_data_random(self, seed, num_fold):
        # all data filtered should be useful
        np.random.seed(seed)
        np.random.shuffle(self.list_gene)
        num_gene = len(self.list_gene)
        fold_size = [num_gene//num_fold] * num_fold
        for i in range(num_gene - num_gene//num_fold*num_fold):
            fold_size[np.argmin(fold_size)] += 1
        self.list_fold = []
        start = 0
        for i in range(num_fold):
            self.list_fold.append(self.list_gene[start:start+fold_size[i]])
            start += fold_size[i]

    def get_data_pack(self, gene_list, add_nt=False, add_aa=False):
        batch_size = len(gene_list)
        x_input = [self.dict_seq_codon[gene] for gene in gene_list]
        length = [len(self.dict_seq_codon[gene]) for gene in gene_list]
        if add_nt:
            x_nt = [self.dict_seq_nt[gene] for gene in gene_list]
            x_input = [np.concatenate([x_input[i], x_nt[i].reshape((len(x_nt[i])//3, -1))], axis=1) for i in range(batch_size)]
        if add_aa:
            x_aa = [self.dict_seq_aa[gene] for gene in gene_list]
            x_input = [np.concatenate([x_input[i], x_aa[i]], axis=1) for i in range(batch_size)]

        density = []
        for gene in gene_list:
            temp = self.dict_density[gene].copy()
            # remove 10 codons near 5' and 3'
            temp[:5] = 0
            temp[-4:] = 0
            density.append(temp)
        self.density = np.array(density)
        self.x_input = np.array(x_input)
        self.length = np.array(length)

    def get_data_pack_cls(self, gene_list, threshs):
        num_cls = len(threshs) + 1
        density_level = []
        threshs = [10**i for i in threshs]
        for gene in gene_list:
            temp = self.dict_density[gene].copy()
            levels = np.zeros(len(temp), dtype=int)
            
            index = temp < threshs[0]
            levels[index] = 0
            for level in range(1, num_cls-1):
                index = (temp >= threshs[level-1]) * (temp < threshs[level])
                levels[index] = level
            index = temp >= threshs[-1]
            levels[index] = num_cls - 1

            density_level.append(levels)
        self.density_level = np.array(density_level)

