[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# ECGNN: Enhancing Abnormal Recognition in 12-Lead ECG with Graph Neural Network 

Cite: Zhao X, Liu Z, Han L, et al. ECGNN: Enhancing Abnormal Recognition in 12-Lead ECG with Graph Neural Network[C]//2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2022: 1411-1416.

## Qick started

#### Dataset Download
The `PTB-XL` dataset can be downloaded from the [Physionet website](https://physionet.org/content/ptb-xl/1.0.1/).

The `ICBEB2018` dataset can be downloaded from the [ICBEB Challenge website](http://2018.icbeb.org/Challenge.html).

#### Setting up the environment
- All the development work is done using `Python 3.7`
- Install all the necessary dependencies using `conda-environment.yaml` file. 

#### What each file does

- `ecg_train.py` training and testing
- `gnn` graph neural network module
- `models` contains scripts for each model
- `utils` contains utilities for `ecg_data`,  and `metrics`

#### Training the model
- To run  `python ecg_train.py --data_dir data/ptb --batchsize 32 --epochs 30 --train True --loggr True`

#### Testing the model
- To run  `python ecg_train.py --data_dir data/ptb --test True`

#### Logs and checkpoints
- The logs are saved in `logs/` directory.
- The model checkpoints are saved in `checkpoints/` directory.



