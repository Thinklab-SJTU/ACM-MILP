# ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation

This is the code of paper **"ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation"**. *Ziao Guo, Yang Li, Chang Liu, Wenli Ouyang, Junchi Yan.* ICML 2024. 

## Environment
- Python environment
    - python 3.7
    - pytorch 1.13
    - torch-geometric 2.3
    - ecole 0.7.3
    - pyscipopt 3.5.0
    - community 0.16
    - networkx
    - pandas
    - tensorboardX
    - gurobipy

- MILP Solver
    - [Gurobi](https://www.gurobi.com/) 10.0.1. Academic License.

- Hydra
    - [Hydra](https://hydra.cc/docs/intro/) for managing hyperparameters and experiments.


In order to build the environment, you can follow commands in `scripts/environment.sh`.

Or alternatively, to build the environment from a file,
```
conda env create -f scripts/environment.yml
```

## Usage

Go to the root directory. Put the datasets under the `./data` directory. Below is an illustration of the directory structure.
```
ACM-MILP
├── conf
├── data
│   ├── ca
│   │   ├── train/
│   │   └── test/
│   ├── mis
│   │   ├── train/
│   │   └── test/
│   └── setcover
│       ├── train/
│       └── test/
├── scripts/
├── src/
├── README.md
├── generate.py
├── preprocess.py
└── train.py
```

The hyperparameter configurations are in `./conf/`.
The commands to run for all datasets are in `./scripts/`.
The main part of the code is in `./src/`.
The workflow of ACM-MILP (using MIS as an example) is as following.

### 1. Preprocessing

To preprocess a dataset,
```
python preprocess.py dataset=mis num_workers=10
```
This will produce graph data for instances and the statistics of the dataset to be used for training. The preprocessed results are saved under `./preprocess/mis/`. 

### 2. Training **ACM-MILP**

To train ACM-MILP with default parameters,
```
python train.py dataset=mis cuda=0 num_workers=10 job_name=mis-default
```
The training log is saved under `TRAIN DIR=./outputs/train/${DATE}/${TIME}-${JOB NAME}/`. The model ckpts are saved under `${TRAIN DIR}/model/`. The generated instances and benchmarking results are saved under `${TRAIN DIR}/eta-${eta}/`.

### 3. Generating new instances

To generate new instances with a trained model,
```
python generate.py dataset=mis \
    generator.mask_ratio=0.01 \
    cuda=0 num_workers=10 \
    dir=${TRAIN DIR}
```
The generated instances and benchmarking results are saved under `${TRAIN DIR}/generate/${DATE}/${TIME}`.

## Citation

If you find this code useful, please consider citing the following paper.

```
@inproceedings{
guo2024acmmilp,
title={ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation},
author={Ziao Guo, Yang Li, Chang Liu, Wenli Ouyang, Junchi Yan},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
}
```


