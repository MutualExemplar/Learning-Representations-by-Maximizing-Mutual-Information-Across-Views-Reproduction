# Mutual_Exemplar
A contrastive learning based semi-supervised segmentation network for medical image segmentation
This repository contains the implementation of a novel contrastive learning based semi-segmentation networks to segment the surgical tools.



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/min-max-similarity-a-contrastive-learning/semi-supervised-semantic-segmentation-on-33)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-33?p=min-max-similarity-a-contrastive-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/min-max-similarity-a-contrastive-learning/semi-supervised-semantic-segmentation-on-2017)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-2017?p=min-max-similarity-a-contrastive-learning)

<div align=center><img src="https://github.com/AngeLouCN/Min_Max_Similarity/blob/main/img/architecture.jpg" width="1000" height="450" alt="Result"/></div>
<p align="center"><b>Fig. 1. The architecture of Min-Max Similarity.</b></p>


## Environment

- python==3.6
- packages:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
```
conda install opencv-python pillow numpy matplotlib
```
- Clone this repository
```
git clone https://github.com/AngeLouCN/Min_Max_Similarity
```
## Data Preparation

We use Kvasir dataset to test its performance:
- [Kvasir-instrument](https://datasets.simula.no/kvasir-instrument/)

**File structure**
```
|-- data
|   |-- kvasir
|   |   |-- train
|   |   |   |--image
|   |   |   |--mask
|   |   |-- test
|   |   |   |--image
|   |   |   |--mask


**You can also test on some other public medical image segmentation dataset with above file architecture**

## Usage

- **Training:**
You can change the hyper-parameters like labeled ratio(here used 5%,20% and 50%), batch size, and e.g. in ```train.py```, and directly run the code.

- **Testing:**
You can change the dataset name in ```test.py``` and run the code.

## Segmentation Performance
<div align=center><img src="https://github.com/AngeLouCN/Min_Max_Similarity/blob/main/img/seg_result.jpg" width="650" height="550" alt="Result"/></div>
<p align="center"><b>Fig. 2. Visual comparison of our method with state-of-the-art models. Segmentation results are shown for 50% of labeled training data for Kvasir-instrument, EndVis’17, ART-NET and RoboTool, and 2.4% labeled training data for cochlear implant. From left to right are EndoVis’17, Kvasir-instrument, ART-NET, RoboTool, Cochlear implant and region of interest (ROI) of Cochlear implant. </b></p>


## Citation
```
@article{lou2023min,
  title={Min-Max Similarity: A Contrastive Semi-Supervised Deep Learning Network for Surgical Tools Segmentation},
  author={Lou, Ange and Tawfik, Kareem and Yao, Xing and Liu, Ziteng and Noble, Jack},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```
## Acknowledgement
Our code is based on the [Duo-SegNet](https://github.com/himashi92/Duo-SegNet), we thank their excellent work and repository.
