# Learning Representations by Maximizing Mutual Information Across Views for Medical Image Segmentation - Reproduction

This repository contains the implementation of **Mutual Exemplar**, a novel **contrastive learning-based semi-supervised segmentation network** designed for **medical image segmentation**. The method maximizes **mutual information across multiple views** to enhance feature representation and improve segmentation performance.

<div align="center">
  <img src="https://github.com/MutualExemplar/Learning-Representations-by-Maximizing-Mutual-Information-Across-Views-Reproduction/blob/main/assets/architecture.png" width="1000" height="450" alt="Architecture"/>
</div>
<p align="center"><b>Fig. 1. The architecture of the Mutual Exemplar model.</b></p>

---

## **Paper Detailes**
- **Title:** Learning Representations by Maximizing Mutual Information Across Views for Medical Image Segmentation
- **Authors:** Weihao Weng, Xin Zhu
- **Conference:** MICCAI 2024
- **Paper URL:** [Link to the Paper](https://papers.miccai.org/miccai-2024/paper/0103_paper.pdf)

---

## **Environment Setup**

### **Clone the Repository**
```bash
git clone https://github.com/MutualExemplar/Learning-Representations-by-Maximizing-Mutual-Information-Across-Views-Reproduction.git
cd Learning-Representations-by-Maximizing-Mutual-Information-Across-Views-Reproduction
```
---

- Python 3.9
- Install dependencies using `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
---

## **Dataset Preparation**
The approach has been evaluated on **medical image segmentation dataset**, namely:

- **[Kvasir-instrument](https://datasets.simula.no/kvasir-instrument/)**

### **Dataset Directory Structure**
Ensure that the datasets are stored in the following directory structure:

```
|-- data
|   |-- kvasir
|   |   |-- train
|   |   |   |-- image
|   |   |   |-- mask
|   |   |-- test
|   |   |   |-- image
|   |   |   |-- mask
```

---

## **Usage**

### **1. Training**
Modify hyperparameters such as **labeled ratio** (e.g., 5%, 20%, 50%), **batch size**, in `train.py`.
Then, run:
```bash
python train.py
```

### **2. Testing**
To evaluate the model on different datasets, update the dataset name in `test.py` and run:
```bash
python test.py
```

---

## **References**
- **Papers:**
  1. Lou, A., Tawfik, K., Yao, X., Liu, Z., Noble, J.: Min-max similarity: A contrastive  
     semi-supervised deep learning network for surgical tools segmentation.  
     *IEEE Transactions on Medical Imaging (2023)*
  
  2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,  
     T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style,  
     high-performance deep learning library. *Advances in Neural Information Processing Systems 32 (2019)*

- **Repository** [Min_Max_Similarity](https://github.com/AngeLouCN/Min_Max_Similarity.git)
My work is based on the pre-existing code base of Min_Max_Similarity.

---


