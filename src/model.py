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
    def __init__(self, input_dims, output_dims, RNN_depth, use_cuda=False, num_class=3):
        super(RiboMIMO, self).__init__()
        self.rnn = nn.GRU(input_dims, output_dims, RNN_depth,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(output_dims*2, output_dims),
            nn.ReLU(),
        )
        self.fc_reg = nn.Linear(output_dims, 1)
        self.num_state = num_class + 2
        self.fc_cla = nn.Linear(output_dims, self.num_state)
        self.START = 0
        self.STOP = self.num_state - 1
        self.transition = nn.Parameter(torch.randn(self.num_state, self.num_state))
        self.use_cuda = use_cuda

    def get_rnn_feature(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        return x

    def viterbi(self, emission):
        score = torch.Tensor([-np.inf for _ in range(self.num_state)])
        if self.use_cuda:
            score = score.cuda()
        score[self.START] = 0
        backtrack = []
        for i in range(emission.shape[0]):
            step_score, step_bt = (score + self.transition).max(-1)
            score = step_score + emission[i]
            backtrack.append(step_bt)
        temp_idx = (score + self.transition[self.STOP]).argmax()
        best_score = score[temp_idx]
        best_path = [temp_idx]
        for bptrs_t in reversed(backtrack):
            temp_idx = bptrs_t[temp_idx]
            best_path.append(temp_idx)
        best_path = reversed(torch.LongTensor(best_path)[:-1])
        return best_score, best_path

    def partition(self, emission):
        # Time-consuming
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.num_state), -1e6)
        if self.use_cuda:
            init_alphas = init_alphas.cuda()
        # START_TAG has all of the score.
        init_alphas[0][self.START] = 0.
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # Iterate through the sentence
        for feat in emission:
            forward_var = self.log_sum_exp(forward_var + feat.unsqueeze(-1) + self.transition).unsqueeze(0)
        terminal_var = forward_var + self.transition[self.STOP]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def score_tag(self, emission, tags):
        # Gives the score of a provided tag sequence
        if self.use_cuda:
            tags = torch.cat([torch.tensor([self.START], dtype=torch.long).cuda(), tags])
        else:
            tags = torch.cat([torch.tensor([self.START], dtype=torch.long), tags])
        score = self.transition[tags[1:], tags[:-1]].sum() + emission[torch.arange(len(tags)-1), tags[1:]].sum()
        score = score + self.transition[[self.STOP], tags[-1]]
        return score

    def viterbi_batch(self, emission_list):
        best_score_list, best_path_list = list(zip(*list(map(self.viterbi, emission_list))))
        best_score_list = torch.Tensor(best_score_list)
        return best_score_list, best_path_list

    def partition_batch(self, emission_list):
        list_alpha = list(map(self.partition, emission_list))
        return list_alpha

    def score_tag_batch(self, emission_list, score_tag_list):
        list_score = list(map(self.score_tag, emission_list, score_tag_list))
        return list_score

    def neg_log_likelihood(self, emission_list, score_tag_list):
        score_tag_list = [score_tag_list[i][0:len(emission_list[i])] for i in range(len(score_tag_list))]
        list_score = self.score_tag_batch(emission_list, score_tag_list)
        list_alpha = self.partition_batch(emission_list)
        return -(torch.Tensor(list_score)- torch.Tensor(list_alpha)).mean()

    def forward(self, x, length, level_list):
        x = self.get_rnn_feature(x, length)
        x = self.fc(x)
        y = self.fc_reg(x).squeeze(2)
        z = self.fc_cla(x)
        emission_list = [z[k, 0:length[k].long()] for k in range(z.shape[0])]
        loss = self.neg_log_likelihood(emission_list, level_list)
        return y, z, loss

    @staticmethod
    def log_sum_exp(vec):
        max_score = vec.max(-1)[0]
        return (vec - max_score.unsqueeze(-1)).exp().sum(-1).log() + max_score

    def score_reg(self, pred, label):
        r2 = r2_score(label, pred)
        mse = mean_squared_error(label, pred)
        corr = pearsonr(label, pred)[0]
        return r2, mse, corr

    def score_cls(self, pred, label):
        label = np.eye(self.num_state-1)[label]
        pred = pred[:, 1:self.num_state-2]
        label = label[:, 1:self.num_state-2]
        auroc, aupr = 0.0, 0.0
        try:
            auroc = roc_auc_score(label, pred)
            aupr = average_precision_score(label, pred)
        except:
            pass
        return auroc, aupr


class DataLoader(object):
    """ Data loader for riboseq data"""
    def __init__(self, path, filename):
        self.path = path + "data/"
        self.organism = filename
        self.aa_table = np.array([
            ["Alanine", "Ala", "A"],
            ["Arginine", "Arg", "R"],
            ["Asparagine", "Asn", "N"],
            ["Aspartic acid", "Asp", "D"],
            ["Cysteine", "Cys", "C"],
            ["Glutamine", "Gln", "Q"],
            ["Glutamic acid", "Glu", "E"],
            ["Glycine", "Gly", "G"],
            ["Histidine","His", "H"],
            ["Isoleucine", "Ile", "I"],
            ["Leucine", "Leu", "L"],
            ["Lysine", "Lys", "K"],
            ["Methionine", "Met", "M"],
            ["Phenylalanine", "Phe", "F"],
            ["Proline", "Pro", "P"],
            ["Serine", "Ser", "S"],
            ["Threonine", "Thr", "T"],
            ["Tryptophan", "Trp", "W"],
            ["Tyrosine", "Tyr", "Y"],
            ["Valine", "Val", "V"],
            ["STOP", "Stp", "*"]])
        self.codon2aa = {
            'TTT': 'F',
            'TTC': 'F',
            'TTA': 'L',
            'TTG': 'L',

            'TCT': 'S',
            'TCC': 'S',
            'TCA': 'S',
            'TCG': 'S',

            'TAT': 'Y',
            'TAC': 'Y',
            'TAA': '*',
            'TAG': '*',

            'TGT': 'C',
            'TGC': 'C',
            'TGA': '*',
            'TGG': 'W',

            'CTT': 'L',
            'CTC': 'L',
            'CTA': 'L',
            'CTG': 'L',

            'CCT': 'P',
            'CCC': 'P',
            'CCA': 'P',
            'CCG': 'P',

            'CAT': 'H',
            'CAC': 'H',
            'CAA': 'Q',
            'CAG': 'Q',

            'CGT': 'R',
            'CGC': 'R',
            'CGA': 'R',
            'CGG': 'R',

            'ATT': 'I',
            'ATC': 'I',
            'ATA': 'I',
            'ATG': 'M',

            'ACT': 'T',
            'ACC': 'T',
            'ACA': 'T',
            'ACG': 'T',

            'AAT': 'N',
            'AAC': 'N',
            'AAA': 'K',
            'AAG': 'K',

            'AGT': 'S',
            'AGC': 'S',
            'AGA': 'R',
            'AGG': 'R',

            'GTT': 'V',
            'GTC': 'V',
            'GTA': 'V',
            'GTG': 'V',

            'GCT': 'A',
            'GCC': 'A',
            'GCA': 'A',
            'GCG': 'A',

            'GAT': 'D',
            'GAC': 'D',
            'GAA': 'E',
            'GAG': 'E',

            'GGT': 'G',
            'GGC': 'G',
            'GGA': 'G',
            'GGG': 'G'
        }
        self.aa2codon = {}
        for codon in self.codon2aa.keys():
            if self.codon2aa[codon] not in self.aa2codon:
                self.aa2codon[self.codon2aa[codon]] = {codon}
            else:
                self.aa2codon[self.codon2aa[codon]].add(codon)
        # construct onehot encoding
        self.load_onehot()

        # load data from fasta
        self.load_data()

    def load_onehot(self):
        nts = "ATCG"
        codon2idx = {}
        for nt1 in nts:
            for nt2 in nts:
                for nt3 in nts:
                    codon2idx[nt1+nt2+nt3] = len(codon2idx)
        self.onehot_nt = {nts[i]:np.eye(4)[i] for i in range(len(nts))}
        self.onehot_codon = {codon:np.eye(64)[codon2idx[codon]] for codon in codon2idx}
        self.codon2idx = codon2idx
        self.nt2idx = {nts[i]:i for i in range(4)}

    def load_data(self):
        with open(self.path + self.organism, "r") as f:
            data = f.readlines()
        assert len(data)%3 == 0
        list_name = []
        list_seq = []
        list_density = []
        list_criteria = []
        for i in range(len(data)//3):
            name = data[3*i+0].split('>')[1].split()[0]
            seq = data[3*i+1].split()
            count = [float(e) for e in data[3*i+2].split()]
            avg = np.mean(np.array(count)[np.array(count)>0.5])
            #avg = np.median(np.array(count)[np.array(count)>0.5])
            density = (np.array(count)>0.5) * np.array(count) / avg
            # criteria containing ribosome density AND coverage percentage
            list_criteria.append([np.sum(count), np.mean(count), np.sum(np.array(count)>0.5), np.mean(np.array(count)>0.5)])
            list_name.append(name)
            list_seq.append(seq)
            list_density.append(density)

        list_name = np.array(list_name)
        list_seq = np.array(list_seq)
        list_density = np.array(list_density)
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
        # default
        self.list_gene = self.list_gene_coverage_6

        self.dict_seq = {}
        self.dict_seq_nt = {}
        self.dict_seq_codon = {}
        self.dict_density = {}
        index = []
        for i in range(len(self.list_gene_all)):
            codons = list_seq[i]
            nts = "".join(codons)
            if "N" in nts:
                continue
            index.append(i)
            self.dict_seq[self.list_gene_all[i]] = nts
            self.dict_seq_nt[self.list_gene_all[i]] = np.array([self.onehot_nt[nt] for nt in nts])
            self.dict_seq_codon[self.list_gene_all[i]] = np.array([self.onehot_codon[codon] for codon in codons])
            self.dict_density[self.list_gene_all[i]] = list_density[i]
        self.list_gene_all = self.list_gene_all[index]

        logging.info("{} genes left after filtering in {}".format(len(self.list_gene_all), self.organism))

    def split_data(self, seed, num_fold):
        # only top 500 genes are used?
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

    def position_encoding(self, sentence_size, embedding_size):
        """
        Position Encoding
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        return np.transpose(encoding)

    def get_data_pack(self, gene_list, add_nt=False, add_pos=False):
        batch_size = len(gene_list)
        x_input = [self.dict_seq_codon[gene] for gene in gene_list]
        length = [len(self.dict_seq_codon[gene]) for gene in gene_list]
        if add_nt:
            x_nt = [self.dict_seq_nt[gene] for gene in gene_list]
            x_input = [np.concatenate([x_input[i], x_nt[i].reshape((len(x_nt[i])//3, -1))], axis=1) for i in range(batch_size)]
        if add_pos:
            pos_enc = [self.position_encoding(length[i], 4) for i in range(batch_size)]
            x_input = [np.concatenate([x_input[i], pos_enc[i]], axis=1) for i in range(batch_size)]

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

    def get_data_pack_cls(self, gene_list, threshs, num_cls=3):
        density_level = []
        threshs = [10**i for i in threshs]
        for gene in gene_list:
            temp = self.dict_density[gene].copy()
            levels = np.zeros(len(temp), dtype=int)

            if num_cls == 2:
                index = temp < threshs[1]
                levels[index] = 1
                index = temp >= threshs[1]
                levels[index] = 2
            else:
                index = temp < threshs[0]
                levels[index] = 1
                index = (temp >= threshs[0]) * (temp < threshs[1])
                levels[index] = 2
                index = temp >= threshs[1]
                levels[index] = 3

            density_level.append(levels)
        self.density_level = np.array(density_level)



