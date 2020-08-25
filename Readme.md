# RiboMIMO

This is the source code for paper entitled "Full-length Ribosome Density Prediction by a Multi-Input and Multi-Output Model". 

# Requirement
RiboMIMO is implemented in Python 3, and the requirements are listed as follows:
```
python   3.7.4
pytorch  1.5.0
sklearn  0.21.3
cuda     10.2
```

# Data preparison
Processed ribosome profiling data is needed for training, and the input file should be in fasta format. 
Each sample contains three lines, including:
```
gene name or notation start with ">";
list of codons seperated with tab;
list of counts for ribosome footprints at ribosomal A site.
```
The processed datasets including Subtelny14 (yeast, GSM1289257), Santos19-replete (yeast, GSM3561535), Santos19-starved (yeast, GSM3561537) and Mohammad16 (ecoli, GSE72899), Mohammad19-1 (ecoli, GSM3358140) and Mohammad19-2 (ecoli, GSM3358142) can be found in 
```
data/[DATASET]/
```

# Run
The RiboMIMO model can be trained and cross-validated with command:
```
python -u runRiboMIMO.py --nt --aa --data [DATASET] --gpu [GPU ID] 
```

For more information, please run help:
```
python -u runRiboMIMO.py --help
```

The program will create a working directory and the best models will be stored in the following folder:
```
results/[DATASET]/RiboMIMO_*/
```

# Contact
If you have any questions, please feel free to contact me.

Email: tiantz17@mails.tsinghua.edu.cn
