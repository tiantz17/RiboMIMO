import os
import sys
import json
import argparse
from argparse import Namespace
from runRiboMIMO import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152


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
    model_path = sys.argv[2]
    config = json.load(open(model_path + "config", "r"))
    args = Namespace(**config)
    args.info = "cis"
    args.gpu = sys.argv[1]

    """ claim class instance """
    RiboMIMONet = RiboMIMOModel(args)
    np.random.seed(RiboMIMONet.seed)
    seeds = np.random.randint(0, 1024, RiboMIMONet.num_repeat)
    logging.info("{} repeat {} fold cross-validation".format(RiboMIMONet.num_repeat, RiboMIMONet.num_fold))
    repeat = 0
    RiboMIMONet.split_data(seeds[repeat])
    # replace the model_path with the path containing trained models
    path_save = RiboMIMONet.save_path

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

            ##############
            # for debug only
            ##############
            break
            ##############
    
if __name__ == "__main__":
    print("Usage: python -u getSaliency.py [GPU ID] [MODEL PATH]")
    get_saliency_map()
    print("Done!")
    print("The obtained saliency map matrix CIS[j, i] indicating the contribution from codon i to the ribosome density at codon j")
    