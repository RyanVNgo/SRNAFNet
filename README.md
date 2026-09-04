# SRNAFNet

An adaptation of NAFNet for Single-Image Super-Resolution


## Requirements

- `python 3.12+`
- `pip`


## Setup

Clone the repository:

```bash
git clone https://github.com/RyanVNgo/SRNAFNet.git
cd SRNAFNet
```

Create a python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate # Windows: venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

There are three scripts available for use: `train.py`, `eval.py`, and `demo.py`. Each
script is found in the `SISR` directory. I suggest running the scripts with the `-h`
command line argument to understand the needed command line arguments to control each
script.

Example usage of `demo.py` script:
```bash
python SISR/demo.py -m pretrained_models/model.pth -i /path/to/input.png -o /path/to/output.png
```


## Citation

This project is based in part on the research from:

Chen, Liangyu, Chu, Xiaojie, Zhang, Xiangyu, and Sun, Jian.  
*Simple Baselines for Image Restoration*. arXiv:2204.04676, 2022.  
https://arxiv.org/abs/2204.04676

Please cite those mentioned if you use this project in your application/research.


