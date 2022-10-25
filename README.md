# Expressive Graph Decision Networks
Authors: **Matthew Morris, Thomas D. Barrett, Arnu Pretorius**

This is the code used in our paper, ["Universally Expressive Communication in Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/2206.06758), which is published in [NeurIPS 2022](https://nips.cc/Conferences/2022).

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

## Baselines
We augment 6 existing successful communication baselines with RNI and Unique IDs:
- CommNet
- IC3Net
- TarMAC
- TarMAC-IC3Net
- MAGIC
- DGN

## Environments
- Traffic Junction (communication benchmark)
- Predator Prey (communication benchmark)
- Box Pushing (new, designed to test communication expressivity beyond 1-WL)
- Drone Scatter (new, designed to test ability to perform symmetry breaking)

## Reproducing Results
To reproduce the exact results shown in our paper, use
the hyperparameters found in our appendix.

## Citation
If you find our paper or code helpful to your research, please consider citing the paper:
```
@inproceedings{morris2022universally,
  title={Universally Expressive Communication in Multi-Agent Reinforcement Learning},
  author={Morris, Matthew and Barrett, Thomas D and Pretorius, Arnu},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS)},
  year={2022}
}
```

## Reference
We adapted the training framework from [MAGIC](https://github.com/CORE-Robotics-Lab/MAGIC) and used their implementations as starting points for most of our baselines. We used [pytorch_DGN](https://github.com/jiechuanjiang/pytorch_DGN) as a starting point for our implementation of DGN.
