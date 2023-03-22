# Grad-Attacks

This repository contains the code used in our study on **Evaluating Vulnerabilities of Deep Learning Explainability for
Medical Images in Adversarial Settings**


## Usage

The codes are splited into two directories, scripts_evaluate aim to evaluate images, as well as scripts_experimentos to trian models and execute attacks.

### 1. Requeriments

- Python 3.8
- Tensorflow 2.10
- adversarial-robustness-toolbox
- numpy
- Pandas

### 2. Configure environment

**2.1. Create environment using conda and set up tensorflow and python** 
```shell
  conda create -n cbms tensorflow=2.10 python=3.8
```
```shell
  conda activate cbms
```

**2.2. Install libraries**

```shell
  pip install -r requeriments.txt
```

### 2. Download the following datasets

- Chest X-ray for PNEUMONIA: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- OCT: https://www.kaggle.com/datasets/paultimothymooney/kermany2018ls

### 3. Train models

**Parameters:** 
  - databaset_csv: chest_xray or OCT
  - model_name: inceptionv3, resnet, or vgg16

```shell
python3 train_models.py --database_csv [database] --model_name [model name]
```
p.s.: the database csv should be saved with name of respectively dataset, such as chest_xray.csv or OCT.csv

### 4. test models

test models to extract metrics in test phase, such as accuracy, AUC, and Fooling Rate

**Parameters:** 
  - databaset_csv: chest_xray or OCT
  - model_name: inceptionv3, resnet, or vgg16

```shell
python3 train_models.py --database_csv [database] --model_name [model name]
```

### 5. Attack models on test phase

**5.1** Install the library to generate attacks
- ```pip install adversarial-robustness-toolbox ```

**5.2** Generate attacks

Params:

### 6. Generate grad-cam



### 7. Evaluation

**7.1** Measure distortion

**7.1** Measure attack strenghs

### 8. Results
