# Grad-Attacks

This repository contains the code used in our study on **Evaluating Vulnerabilities of Deep Learning Explainability for
Medical Images in Adversarial Settings**

**Authors**: [Erikson J. de Aguiar](https://github.com/eriksonJAguiar), [Márcus V. L. Costa](https://github.com/usmarcv), Caetano Traina Jr. and Agma J. M. Traina

**Abstract:**
Deep Learning (DL) incorporates a useful set of techniques and improves medical diagnosis based on imaging exams, such as Chest X-rays (CXR) and optical coherence tomography (OCT). However, DL models can be biased, so explainability can help guide the decision-making process, including the well-known method Gradient-weighted Class Activation Mapping (GRAD-CAM). Moreover, DL models suffer from adversarial attacks (ADV) through creation of perturbed examples, which decrease the model’s confidence. In this paper, we evaluate DL vulnerabilities of DL applied to medical images and analyze the effects of attacks on GRAD-CAM heatmaps, aimed at verifying the strengths of attack. We evaluated our technique using ResNet50, VGG16, and InceptionV3 on CXR and OCT datasets. Our experiments evaluated the Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and DeepFool, both on  clean and attacked datasets. We measured the Fooling Rate (FR), image degradation, and attack strengths, quantifying the influence of attacks in the model based on Grad-CAM analysis. The experiments showed that DeepFool presented the strongest FR, achieving up to 93% for CXR and 96% for OCT. However, PGD obtained the best malicious deeds for multi-class (OCT), whereas DeepFool was the most hurtful for binary class (CXR) when it evaluated image degradation and attack strengths. DeepFool produces a visual effect that is hard to notice on Grad-CAM, which is more suitable for attacker deeds. Our analysis can be used to understand ADV most likely to be used by an attacker over medical images and their effects on explainability.

**Proposal overview**

![](worflow.pdf)

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

### 3. Download the following datasets

- Chest X-ray for PNEUMONIA: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- OCT: https://www.kaggle.com/datasets/paultimothymooney/kermany2018ls

### 4. Train models

**Parameters:** 
  - databaset_csv: chest_xray or OCT
  - model_name: inceptionv3, resnet, or vgg16

```shell
python3 train_models.py --database_csv [database] --model_name [model name]
```
p.s.: the database csv should be saved with name of respectively dataset, such as chest_xray.csv or OCT.csv

### 5. test models

test models to extract metrics in test phase, such as accuracy, AUC, and Fooling Rate

**Parameters:** 
  - databaset_csv: chest_xray or OCT
  - model_name: inceptionv3, resnet, or vgg16
  - eps: 0, 0.01, 0.05, 0.075, and 0.1
  
if eps is 0 images tested are clean, on the other hand, without attack.

```python
python3 evaluate_attacked_images.py --database [database] --model_name [model name] --eps [noise level]
```

### 6. Generate and save attacked images

We used attacks FGSM, PGD, and DeepFool.

**Parameters:** 
  - databaset_csv: chest_xray or OCT
  - model_name: inceptionv3, resnet, or vgg16
  - attack_name: fgsm, pgd, or deep

```python
python3 generate_attacked_images.py --database_csv [database] --model_name [model name] --attack_name [attack name]
```

### 7. Generate grad-cam

**Parameters:** 
  - attack_db: path attacked
  - original: path original
  
```python
python3 gradcam_generate.py --attack_db [database attacked path] --original [database original path]
```

### 8. Evaluation

**8.1** Measure distortion

Measure PSRN and SSMI metrics

**Parameters:** 
  - database: database original path
  - attack_path: database attacked path
  - eps: noise level [0.01, 0.05, 0.075, and 0.1]
  
```python
python3 measure_distortion.py --attack_db [database attacked path] --original [database original path]
```
**8.2** Measure attack strengths

Measure NISSMI and MOD.

**Parameters:** 
  - database: grad-cam heatmaps path
  
```python
python3 measure_gradcam.py --database [database grad-cam]
```

### 9. Results
