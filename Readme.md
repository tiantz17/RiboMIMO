# RiboMIMO

This is the source code for paper entitled "Full-length Ribosome Density Prediction by a Multi-Input and Multi-Output Model". 

To reproduce the results in our paper or analyze new ribo-seq dataset using RiboMIMO, you need to follow the steps below:

# 0. Requirement
RiboMIMO is implemented in Python 3, and the requirements are listed as follows:
```
python   3.7.4
pytorch  1.5.0
sklearn  0.21.3
cuda     10.2
```

# 1. Prepare data
Processed ribosome profiling data is needed for training, and the input file should be in fasta format. 
Each sample contains three lines, including:
- gene name or notation start with ">";
- list of codons seperated with tab;
- list of counts for ribosome footprints at ribosomal A site.

For example,
```
>YKL096W-A 93
ATG 	CAA 	TTC 	TCT 	ACT 	GTC 	GCT 	TCC 	GTT 	GCT 	TTC 	GTC 	GCT 	TTG 	GCT 	AAC 	TTT 	GTT 	GCC 	GCT 	GAA 	TCC 	GCT 	GCC 	GCC 	ATT 	TCT 	CAA 	ATC 	ACT 	GAC 	GGT 	CAA 	ATC 	CAA 	GCT 	ACT 	ACC 	ACT 	GCT 	ACC 	ACC 	GAA 	GCT 	ACC 	ACC 	ACT 	GCT 	GCC 	CCA 	TCT 	TCC 	ACC 	GTT 	GAA 	ACT 	GTT 	TCT 	CCA 	TCC 	AGC 	ACC 	GAA 	ACT 	ATC 	TCT 	CAA 	CAA 	ACT 	GAA 	AAT 	GGT 	GCT 	GCT 	AAG 	GCC 	GCT 	GTC 	GGT 	ATG 	GGT 	GCC 	GGT 	GCT 	CTA 	GCT 	GCT 	GCT 	GCT 	ATG 	TTG 	TTA 	TAA
0 	1995 	399 	175 	224 	377 	195 	219 	283 	230 	965 	1188 	425 	482 	305 	89 	677 	531 	395 	813 	190 	171 	216 	339 	982 	236 	150 	485 	214 	1083 	1282 	2416 	1415 	859 	912 	1058 	959 	532 	621 	1524 	304 	158 	191 	169 	199 	95 	149 	235 	122 	76 	50 	153 	28 	82 	252 	141 	213 	344 	599 	1847 	343 	189 	139 	84 	95 	195 	475 	464 	968 	656 	621 	1271 	723 	422 	224 	1325 	333 	59 	652 	1388 	76 	236 	168 	374 	686 	420 	378 	592 	369 	0 	0 	0 	0
```

The already processed datasets including Subtelny14 (yeast, GSM1289257), Mohammad16 (ecoli, GSE72899), Mohammad19-1 (ecoli, GSM3358140) and Mohammad19-2 (ecoli, GSM3358142) can be found in 
```
data/[DATASET].txt
```

You can also use customed datset following the same fasta format, name it ```[DATASET].txt```, and put it in the ```data/``` folder.

# 2. Train RiboMIMO
The RiboMIMO model can be trained and cross-validated with command:
```
cd src/
python -u runRiboMIMO.py --add_nt --add_aa --dataset [DATASET] --gpu [GPU ID] 
```

For more information, please run help:
```
python -u runRiboMIMO.py --help
```

The program will create a working directory and the best models will be stored in the following ```[MODEL FOLDER]```:
```
results/RiboMIMO_[DATASET]*/
```

# 3. Compute CIS
To analyze the trained RiboMIMO model, we use saliency maps to account for the codon contributions, termed codon impact score (CIS).

CIS can be calculated using the trained RiboMIMO model with command:
```
cd src/
python -u getSaliency.py [GPU ID] [MODEL FOLDER]
```

The program will create a working directory and the CIS results will be stored in the following ```[CIS FOLDER]```:
```
results/CIS_[DATASET]*/
```

Note that the folder name of ```[CIS FOLDER]``` follows the same pattern with ```[MODEL FOLDER]```.

# 4. Analyze
The code for analyzing the trained models and CIS is provided in jupyter notebook ```src/analyses.ipynb```.

The program will create a working directory and the analysis results will be stored in the following ```[ANALYSIS FOLDER]```:
```
results/Analysis_[DATASET]*/
```

# 5. Predictions
The predicted ribosome densities at each codon position of genes in the test sets are provided in the ```prediction/```.

The results are stored in ```dict``` format, and can be loaded using the following python command:
```
import pickle
# load dict
predict_dict = pickle.load(open("./prediction/RiboMIMO_yeast_Subtelny14_pred", "rb"))
# get label and prediction results for gene YOR098C
label, pred = temp['YOR098C']
```



# Contact
If you have any questions, please feel free to contact me.

Email: tiantz17@mails.tsinghua.edu.cn
