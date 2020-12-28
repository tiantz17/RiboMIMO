import os, sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.stats import spearmanr
from runRiboMIMO import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import argparse

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

# Fast!
def get_saliency_one_sample(RiboMIMONet, idx, len_codon):
    # calculate saliency map
    saliency_sum = []
    x_input, length, density, mask_density, level, batch_index = RiboMIMONet.get_data_batch(np.array([idx]))
    x_input.requires_grad = True 
    RiboMIMONet.model.zero_grad()
    x_input.grad = None      
    pred = RiboMIMONet.model.predict(x_input, length)
    for jdx in range(len_codon):
        # print("{:.2f}%\r".format((jdx+1.0)/len_codon*100), end="")    
        RiboMIMONet.model.zero_grad()
        x_input.grad = None
        pred[0, jdx].backward(retain_graph=True)
        saliency = x_input.grad.cpu().data * x_input.cpu().data
        saliency_sum.append(saliency.clone())
    saliency_sum = torch.cat(saliency_sum)
    saliency_full = saliency_sum.numpy().sum(2)
    return saliency_full

def get_saliency_map():
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    args = parser.parse_args([])

    args.nt = True
    args.aa = True

    args.info = "test"
    args.dataset = sys.argv[2]
    """ claim class instance """
    RiboMIMONet = RiboMIMOModel(args)
    print(RiboMIMONet.dataset)
    np.random.seed(RiboMIMONet.seed)
    seeds = np.random.randint(0, 1024, RiboMIMONet.num_repeat)
    logging.info("{} repeat {} fold cross-validation".format(RiboMIMONet.num_repeat, RiboMIMONet.num_fold))
    repeat = 0
    RiboMIMONet.split_data(seeds[repeat])
    # replace the model_path with the path containing trained models
    model_path = sys.argv[3]

    for fold in range(args.num_fold):
        logging.info("{}/{} repeat {}/{} fold".format(repeat+1, RiboMIMONet.num_repeat, fold+1, RiboMIMONet.num_fold))
        RiboMIMONet.load_model()
        RiboMIMONet.model.load_state_dict(torch.load(model_path+"best_model_{}_{}.pth".format(repeat, fold)))
        RiboMIMONet.index_test = RiboMIMONet.data.list_fold[fold]
        RiboMIMONet.index_train = RiboMIMONet.data.list_fold[:fold] + RiboMIMONet.data.list_fold[fold+1:]
        RiboMIMONet.index_train = np.concatenate(RiboMIMONet.index_train)
        index_eval = RiboMIMONet.index_test
        RiboMIMONet.get_thresh()
        RiboMIMONet.data.get_data_pack(index_eval, RiboMIMONet.add_nt, RiboMIMONet.add_aa)
        RiboMIMONet.data.get_data_pack_cls(index_eval, RiboMIMONet.threshs)
        path_save = "./results/CIS_"+args.dataset+"/"
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        for idx in range(len(index_eval)):
            gene_name = index_eval[idx]
            seq = RiboMIMONet.data.dict_seq[gene_name]
            len_codon = len(seq)//3
            x_input, length, density, mask_density, level, batch_index = RiboMIMONet.get_data_batch(np.array([idx]))
            pred, pred_cls, loss = RiboMIMONet.model(x_input, length)
            print(idx, gene_name, len_codon)
            print(pearsonr(pred[0].cpu().data.numpy(), density[0].cpu().data.numpy())[0])

            saliency_full = get_saliency_one_sample(RiboMIMONet, idx, len_codon)    
            pickle.dump(saliency_full, open(path_save+"{}".format(gene_name), "wb"))
    
if __name__ == "__main__":
    print("Usage: python -u getSaliency.py [GPU ID] [DATASET] [MODEL PATH]")
    print("The obtained saliency map matrix CIS[j, i] indicating the contribution from i to j")
    get_saliency_map()
