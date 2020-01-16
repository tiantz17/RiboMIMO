# RiboMIMO: Full-length Ribosome Distribution Prediction by a Multi-Input Multi-Output Model
<img src="data/model.png" width="600" align="center">

This is the source code for paper entitled "RiboMIMO: Full-length Ribosome Distribution Prediction by a Multi-Input Multi-Output Model". 

# Requirement
RiboMIMO is implemented in Python 3, and the requirements are listed as follows:
```
python   3.7.4
pytorch  1.1.0
sklearn  0.21.3
```

# Data preparing
Processed ribosome profiling data is needed for training, and the input file should be in fasta format. 
Each sample contains three lines, including:
```
gene name or notation start with ">";
codon list seperated with tab;
counts of ribosome footprints at ribosomal A site.
```
An example of data format can be found in
```
data/example
```

# Run
The RiboMIMO model can be trained and cross-validated with command:
```
python -u runRiboMIMO.py --data [FILENAME] --gpu [GPU ID] 
```

For more information, please run help:
```
python -u runRiboMIMO.py --help
```

The program will create a working directory and the best models will be stored in the following folder:
```
results/RiboMIMO_*/
```

If you have any questions, please feel free to contact me
Email: tiantz17@mails.tsinghua.edu.cn