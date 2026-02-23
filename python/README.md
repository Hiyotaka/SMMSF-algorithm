# Python Implementation

This folder provides a Python port of the MATLAB SMMSF algorithm.

## Install

```bash
pip install -r requirements.txt
```

## Run From CSV

```bash
python python/run_smmsf.py --input data.csv --k 4 --output labels.csv
```

## Import As Function

```python
import numpy as np
from python.smmsf import smmsf_clustering

x = np.loadtxt("data.csv", delimiter=",")
labels = smmsf_clustering(x, k=4)
```

Notes:
- Labels are 1-based to stay MATLAB-compatible.
- Input should be numeric and shaped `(n_samples, n_features)`.
