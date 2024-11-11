# <div align="center">:dolphin:DOLPHIN</div>

<div align="center">
  <a href="http://dlvc-lab.net/lianwen/"> <img alt="SCUT DLVC Lab" src="https://img.shields.io/badge/SCUT-DLVC_Lab-blue?logo=Academia&logoColor=hsl"></a>
  <a href="https://github.com/SCUT-DLVCLab/DOLPHIN"> <img alt="DOLPHIN Project" src="https://img.shields.io/badge/DOLPHIN-Deep_Learning-green?logo=github&logoColor=hsl&"></a>
  <img alt="Static Badge" src="https://img.shields.io/badge/Pytorch-orange?logo=pytorch&logoColor=rgb">
  <img alt="Static Badge" src="https://img.shields.io/badge/Python-lightgray?logo=python&logoColor=rgb">


Official code of the [DOLPHIN](https://ieeexplore.ieee.org/document/10746457) model (TIFS 2024) and the release of the OLIWER dataset.
</div>

## :earth_asia:Environment

```bash
git clone https://github.com/SCUT-DLVCLab/DOLPHIN.git
conda create -n dolphin python=3.8.16
conda activate dolphin
pip install -r requirements.txt
```

## :hammer_and_pick:Data Preparation

Download the three subsets: CASIA-OLHWDB2, DCOH-E, and SCUT-COUCH2009 using the following links:

- [Baidu Cloud](https://pan.baidu.com/s/1Op917v5IM7OushQ_xPNLSg?pwd=oler)
- [Google Drive](https://drive.google.com/drive/folders/1W-R78wLSJXDhK998c_zIAEFxtPE10AX4?usp=sharing)

Create a local directory named `data-raw` Unzip the .zip archives using the following commands:

```bash
unzip OLHWDB2.zip -d .
unzip DCOH-E.zip -d .
unzip COUCH09.zip -d .
```

The directory should look like this:

```
data-raw
├── COUCH09
│   ├── 001
│   └── ...
├── DCOH-E
│   ├── dcoh-e313
│   └── ...
└── OLHWDB2
    ├── 001
    └── ...
```

Then run `preprocess.py` for data preprocessing:

```bash
python preprocess.py --dataset olhwdb2
python preprocess.py --dataset dcohe
python preprocess.py --dataset couch
```

The preprocessed data will be saved at the `data` folder.

Then run the `divide.py` to merge the three subsets into the **OLIWER** dataset and divide the data into `training` and `testing` parts.

```bash
python divide.py --divide
python divide.py --extract
```

Now the data should be all preprocessed. The final data directory should look like:

```bash
data
├── COUCH09
│   └── COUCH09.pkl
├── DCOH-E
│   └── DCOH-E.pkl
├── OLHWDB2
│   └── OLHWDB2.pkl
└── OLIWER
    ├── split.json
    ├── test.pkl
    ├── test-tf.pkl
    ├── train.pkl
    └── train-tf.pkl
```

## :rocket:Test

```
python test.py --weights weights/model.pth
```

## :bookmark_tabs:Citation

```
@ARTICLE{10746457,
  author={Zhang, Peirong and Jin, Lianwen},
  journal={IEEE Transactions on Information Forensics and Security (TIFS)}, 
  title={{Online Writer Retrieval with Chinese Handwritten Phrases: A Synergistic Temporal-Frequency Representation Learning Approach}}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
 }
```

## :phone:Cotact

Peirong Zhang: eeprzhang@mail.scut.edu.cn

## :palm_tree:Copyright

Copyright 2024, Deep Learning and Vision Computing (DLVC) Lab, South China China University of Technology. [http://www.dlvc-lab.net](http://www.dlvc-lab.net/).

