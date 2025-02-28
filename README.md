# Learning Representations by Maximizing Mutual Information Across Views for Medical Image Segmentation

This repository contains the implementation of **Mutual Exemplar**, a novel **contrastive learning-based semi-supervised segmentation network** designed for **medical image segmentation**. Our method maximizes **mutual information across multiple views** to enhance feature representation and improve segmentation performance.

<div align="center">
  <img src="https://github.com/MutualExemplar/Learning-Representations-by-Maximizing-Mutual-Information-Across-Views-Reproduction/blob/main/assets/architecture.png" width="1000" height="450" alt="Architecture"/>
</div>
<p align="center"><b>Fig. 1. The architecture of the Mutual Exemplar model.</b></p>

---

## **Environment Setup**

- Python 3.6  
- Install dependencies using `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### **Clone the Repository**
```bash
git clone https://github.com/MutualExemplar/Learning-Representations-by-Maximizing-Mutual-Information-Across-Views-Reproduction.git
cd Learning-Representations-by-Maximizing-Mutual-Information-Across-Views-Reproduction
```

---

## **Dataset Preparation**
Our approach has been evaluated on **multiple medical image segmentation datasets**, including:

- **[Kvasir-instrument](https://datasets.simula.no/kvasir-instrument/)**
- **[EndoVis’17](https://endovissub-instrument.grand-challenge.org/)**
- **[ART-NET](https://www.sciencedirect.com/science/article/pii/S1361841521000861)**
- **[RoboTool](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_17)**
- **[FEES](https://fees-challenge.grand-challenge.org/)**

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
Modify hyperparameters such as **labeled ratio** (e.g., 5%, 20%, 50%), **batch size**, and other training parameters in `train.py`.  
Then, run:
```bash
python train.py
```

### **2. Testing**
To evaluate the model on different datasets, update the dataset name in `test.py` and run:
```bash
python test.py
```



