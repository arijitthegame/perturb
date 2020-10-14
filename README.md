#Perturb-seq

## Rrequirements 
- scanpy 
- numpy as np
- scipy as sp
- pandas as pd
- json
- os
- matplotlib
- seaborn 
- rpy2
- logging
- anndata2ri
- torch
- time
- gflags
- collections
- sys
- pickle
- random
- scipy
- sklearn

## Preprocessing steps

- Make the scanpy data with annotation of perterbations
- Quality control
- Normalization
- Batch Correction
- Highly Variable Genes
- PCA

## run step
- run the preprocessing.ipynb and get total_after.h5ad
- train and test by running

```shell
python3 train.py 
```
