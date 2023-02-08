# Grad-Attacks

This repository contains the code used in our study on **Evaluating Vulnerabilities of Deep Learning Explainability for
Medical Images in Adversarial Settings**


## Usage

The codes are splited into two directories, scripts_evaluate aim to evaluate images, as well as scripts_experimentos to trian models and execute attacks.

### 1. Requeriments

- Python 3.8
- Tensorflow 2
- adversarial-robustness-toolbox
- numpy
- Pandas

### 2. Download the foolowing datasets

- Chest X-ray for PNEUMONIA: 
- OCT: 

  **2.1** Balance datasets
  
```shell
python3 
```

### 3. Train models

**Parameters:** 
  - databaset_csv: chest_xray or OCT
  - model_name: inceptionv3, resnet, or vgg16

```shell
python3 train_models.py --database_csv [database] --model_name [model name]
```

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
