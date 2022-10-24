# Expressive GDNs
**Matthew Morris, Thomas D. Barrett, Arnu Pretorius**

This is the code used in our paper, [Universally Expressive Communication in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2206.06758), which was published in NeurIPS 2022.

## Installation
`make build` can be used to create a Docker image
with all the required packages installed. `make bash`
can then be used to open a terminal within the created 
image.

Alternatively, create a virtual environment and then
install the requirements manually:

```
python -m pip install -r requirements.txt
cd src/envs/ic3net-envs
python setup.py develop
```

## Running Experiments
`main.py` is the central runner script. The `scripts`
folder contains examples for how to run different models
on various environments. For example, to run CommNet on
Easy Traffic Junction, execute:
```
sh scripts/train_tj_easy_commnet.sh
```

## Reproducing Results
To reproduce the exact results shown in our paper, use
the hyperparameters found in our appendix.
